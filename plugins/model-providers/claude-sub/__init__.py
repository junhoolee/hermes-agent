"""Claude Pro/Max/Team subscription provider (claude-agent-sdk plugin).

Drives ``claude-agent-sdk`` directly instead of an ACP subprocess: the SDK
resolves the user's own ``claude auth login`` session, so this plugin never
holds, forwards, or refreshes a credential. It supplies its own client
through :meth:`providers.base.ProviderProfile.create_client` — the same
registration seam ``plugins/model-providers/copilot-acp/`` uses — so nothing
in core needs to change for it to exist.

v0.1-A was a tool-less, one-shot text path; streaming, tool bridging,
session continuation, and (v0.1-G) image input have since been added — see
``client.py``, ``session.py``, and ``convert.py``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from providers import register_provider
from providers.base import ProviderProfile


def _resolve_process_command() -> str:
    """Prefer the SDK's own bundled CLI; fall back to a bare ``claude`` lookup.

    ``claude_agent_sdk`` is an optional extra, imported lazily and guarded so
    this module (and provider registration) works even when it is absent.
    """
    try:
        import claude_agent_sdk

        bundled = Path(claude_agent_sdk.__file__).parent / "_bundled" / "claude"
        if bundled.exists():
            return str(bundled)
    except Exception:
        pass
    return "claude"


class ClaudeSubProfile(ProviderProfile):
    """Claude subscription via claude-agent-sdk — no REST endpoint, no API key."""

    def create_client(self, **client_kwargs: Any) -> Any:
        from .client import ClaudeSubClient

        return ClaudeSubClient(**client_kwargs)

    def build_extra_body(self, *, session_id: str | None = None, **context: Any) -> dict[str, Any]:
        return {"hermes_session_id": session_id} if session_id else {}

    def fetch_models(
        self,
        *,
        api_key: str | None = None,
        base_url: str | None = None,
        timeout: float = 8.0,
    ) -> list[str] | None:
        """No REST models endpoint — the SDK has no live catalog to query."""
        return None


claude_sub = ClaudeSubProfile(
    name="claude-sub",
    aliases=("claude-sdk-sub",),
    display_name="Claude subscription (Agent SDK plugin)",
    description=(
        "Claude Pro/Max/Team via claude-agent-sdk (out-of-tree plugin); "
        "run `claude auth login` first"
    ),
    api_mode="chat_completions",
    auth_type="external_process",
    base_url="claude-sub://sdk",
    env_vars=(),
    supports_health_check=False,
    # This flag means "does this provider accept native images inside a
    # tool-result message" (providers/base.py); this bridge's tool results
    # are flattened to text in client.py::_resolve_pending, so flipping it
    # to True would make tools/vision_tools.py's native fast path silently
    # drop tool-result images. A last-user-turn image (v0.1-G) is delivered
    # via convert.StreamPrompt regardless of this flag.
    supports_vision=False,
    fallback_models=("claude-sonnet-5", "claude-opus-5", "claude-fable-5-1"),
    process_command=_resolve_process_command(),
    process_command_env_vars=("HERMES_CLAUDE_SUB_CLI",),
    process_args=(),
    process_args_env_var="HERMES_CLAUDE_SUB_ARGS",
)

register_provider(claude_sub)
