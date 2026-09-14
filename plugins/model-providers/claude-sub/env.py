"""Sanitized child environment for the claude-sub subprocess.

``ClaudeAgentOptions.env`` cannot express "delete this variable" — the SDK
merges it over a copy of ``os.environ``, so a caller-supplied override can
only ever add or overwrite a key, never remove one. Anthropic/Bedrock/Vertex
credentials or a Hermes session id in the parent's environment would
therefore leak into the child unless they are stripped in the env dict we
hand the transport. This module owns that scrub — deliberately independent
of ``agent/claude_billing.py`` (a fork-only module this plugin must not
import) so the plugin works standalone, in and out of tree.
"""

from __future__ import annotations

import os
from typing import Any, Mapping

# Exact-name credentials that would outrank/bypass the subscription auth the
# claude-agent-sdk CLI resolves for itself.
BLOCKED_EXACT: frozenset = frozenset(
    {
        "ANTHROPIC_API_KEY",
        "ANTHROPIC_AUTH_TOKEN",
        "ANTHROPIC_TOKEN",
        "ANTHROPIC_AWS_API_KEY",
        "ANTHROPIC_FOUNDRY_API_KEY",
        "ANTHROPIC_FOUNDRY_AUTH_TOKEN",
        "ANTHROPIC_IDENTITY_TOKEN_FILE",
        "CLAUDE_CODE_OAUTH_REFRESH_TOKEN",
        "CLAUDE_CODE_OAUTH_TOKEN_FILE_DESCRIPTOR",
        "CLAUDE_CODE_USE_BEDROCK",
        "CLAUDE_CODE_USE_VERTEX",
        "CLAUDE_CODE_USE_FOUNDRY",
        "CLAUDE_CODE_USE_ANTHROPIC_AWS",
        "CLAUDE_CODE_USE_ANTHROPIC_GOOGLE_CLOUD",
        "CLAUDE_CODE_USE_MANTLE",
        "CLAUDE_CODE_USE_GATEWAY",
        "CLAUDECODE",
        "HERMES_SESSION_ID",
        "HERMES_SESSION_SOURCE",
        "HERMES_SINGLE_QUERY_SESSION",
    }
)

# Prefixes stripped regardless of exact name — kanban/session-scoped state the
# child subprocess has no business seeing.
BLOCKED_PREFIXES: tuple = ("HERMES_KANBAN_", "HERMES_SESSION_", "HERMES_SINGLE_QUERY_")

# Always preserved even if a blocklist rule would otherwise remove them.
# HOME in particular must never be dropped: on macOS it drives the
# login-keychain lookup the CLI needs to find its stored OAuth credentials.
PASS_THROUGH: tuple = ("HERMES_HOME", "HOME", "CLAUDE_CONFIG_DIR", "PATH")


def _is_blocked(key: str) -> bool:
    if key in BLOCKED_EXACT:
        return True
    return any(key.startswith(prefix) for prefix in BLOCKED_PREFIXES)


def scrub_env(source: Mapping[str, str]) -> dict:
    """Return a filtered copy of *source* — never mutates the input."""
    passthrough = {k: source[k] for k in PASS_THROUGH if k in source}
    scrubbed = {k: v for k, v in source.items() if not _is_blocked(k)}
    scrubbed.update(passthrough)
    return scrubbed


def build_child_env(options: Any, *, cwd: str | None = None) -> dict:
    """Build the child environment for a claude-sub subprocess spawn.

    Mirrors the sanctioned merge order: os.environ copy → scrub → apply
    entrypoint markers → merge ``options.env`` → re-apply the blocklist (a
    caller-supplied override must not be able to reintroduce a blocked
    credential) → re-apply PASS_THROUGH (HOME must never be dropped).
    """
    env = scrub_env(os.environ)
    env["CLAUDE_CODE_ENTRYPOINT"] = "sdk-py"

    try:
        from claude_agent_sdk._version import __version__ as sdk_version
    except Exception:
        sdk_version = ""
    if sdk_version:
        env["CLAUDE_AGENT_SDK_VERSION"] = sdk_version

    if not env.get("MCP_TOOL_TIMEOUT"):
        env["MCP_TOOL_TIMEOUT"] = "3600000"

    resolved_cwd = cwd if cwd is not None else getattr(options, "cwd", None)
    if resolved_cwd:
        env["PWD"] = str(resolved_cwd)

    env.update(getattr(options, "env", None) or {})

    # Re-scrub after options.env: a caller-supplied override must not be able
    # to put a higher-precedence credential back into the child.
    env = {k: v for k, v in env.items() if not _is_blocked(k)}
    for key in PASS_THROUGH:
        if key in os.environ and key not in env:
            env[key] = os.environ[key]

    return env


def sanitized_transport_class() -> Any:
    """The ``SubprocessCLITransport`` subclass that spawns from ``build_child_env``.

    Built once per process against the installed SDK; imports the SDK lazily
    so this module remains importable without the optional extra.
    """
    global _TRANSPORT_CLASS
    if _TRANSPORT_CLASS is None:
        _TRANSPORT_CLASS = _build_transport_class()
    return _TRANSPORT_CLASS


_TRANSPORT_CLASS: Any = None


def _build_transport_class() -> Any:
    import anyio
    from anyio.streams.text import TextReceiveStream, TextSendStream
    from subprocess import PIPE

    from claude_agent_sdk._errors import CLIConnectionError, CLINotFoundError
    from claude_agent_sdk._internal.transport.subprocess_cli import (
        _ACTIVE_CHILDREN,
        SubprocessCLITransport,
    )
    from claude_agent_sdk._internal._task_compat import spawn_detached

    class ClaudeSubTransport(SubprocessCLITransport):
        """``SubprocessCLITransport`` that spawns from a scrubbed environment."""

        def build_child_env(self) -> dict:
            return build_child_env(self._options, cwd=self._cwd)

        async def connect(self) -> None:
            if self._process:
                return

            if self._cli_path is None:
                self._cli_path = await anyio.to_thread.run_sync(self._find_cli)
            self._reject_windows_batch_cli(self._cli_path)
            if not os.environ.get("CLAUDE_AGENT_SDK_SKIP_VERSION_CHECK"):
                await self._check_claude_version()

            cmd = self._build_command()
            try:
                process_env = self.build_child_env()
                stderr_dest = PIPE if self._options.stderr is not None else None

                self._process = await anyio.open_process(
                    cmd,
                    stdin=PIPE,
                    stdout=PIPE,
                    stderr=stderr_dest,
                    cwd=self._cwd,
                    env=process_env,
                    user=self._options.user,
                )
                _ACTIVE_CHILDREN.add(self._process)

                if self._process.stdout:
                    self._stdout_stream = TextReceiveStream(self._process.stdout)
                if stderr_dest is PIPE and self._process.stderr:
                    self._stderr_stream = TextReceiveStream(self._process.stderr)
                    self._stderr_task = spawn_detached(self._handle_stderr())
                if self._process.stdin:
                    self._stdin_stream = TextSendStream(self._process.stdin)

                self._ready = True
            except FileNotFoundError as exc:
                if self._cwd and not os.path.exists(self._cwd):
                    error: Exception = CLIConnectionError(
                        f"Working directory does not exist: {self._cwd}"
                    )
                else:
                    error = CLINotFoundError(f"Claude Code not found at: {self._cli_path}")
                self._exit_error = error
                raise error from exc
            except Exception as exc:
                error = CLIConnectionError(f"Failed to start claude-sub CLI: {exc}")
                self._exit_error = error
                raise error from exc

    return ClaudeSubTransport


def build_sanitized_transport(options: Any, *, prompt: Any = None) -> Any:
    """Construct the transport ``ClaudeSDKClient`` should use for *options*."""

    async def _empty_stream():
        return
        yield {}  # pragma: no cover - unreachable, marks this an async generator

    transport_class = sanitized_transport_class()
    return transport_class(
        prompt=prompt if prompt is not None else _empty_stream(),
        options=options,
    )


__all__ = [
    "BLOCKED_EXACT",
    "BLOCKED_PREFIXES",
    "PASS_THROUGH",
    "scrub_env",
    "build_child_env",
    "sanitized_transport_class",
    "build_sanitized_transport",
]
