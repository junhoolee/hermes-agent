"""Shared fixtures for claude-sub plugin tests.

``claude_agent_sdk`` is an optional extra. Every test that needs the plugin's
SDK-facing modules to actually import (``session.py``'s ``build_options``,
``env.py``'s transport builder) uses the ``fake_sdk`` fixture below rather
than the real package, so the suite is deterministic and does not depend on
whether the extra happens to be installed. Not autouse: tests that only
exercise pure-Python helpers (``convert``, ``errors``, ``config``) don't need
it and importing it unconditionally would hide accidental top-level SDK
imports in the plugin's own modules (see AGENTS.md D5 — the SDK must be
imported lazily, inside functions, everywhere in this plugin).
"""

from __future__ import annotations

import importlib
import sys
import types
from dataclasses import dataclass, field
from typing import Any

import pytest


def _load_plugin_module(name: str):
    """Import ``plugins.model_providers.claude_sub.<name>`` after discovery.

    Goes through the real provider-discovery path (not a direct file import)
    so these tests exercise the same loader Hermes uses at runtime.
    ``name="__init__"`` returns the package module itself (the plugin's
    ``__init__.py`` is loaded AS ``plugins.model_providers.claude_sub``, not
    as a ``.__init__`` submodule of it).
    """
    import providers

    providers._discover_providers()
    if name == "__init__":
        return importlib.import_module("plugins.model_providers.claude_sub")
    return importlib.import_module(f"plugins.model_providers.claude_sub.{name}")


@pytest.fixture
def load_plugin_module():
    """Fixture form of :func:`_load_plugin_module`.

    A fixture rather than a plain importable helper: this test directory has
    no ``__init__.py`` (AGENTS.md D13), so a cross-file ``from conftest
    import ...`` would depend on pytest's bare-module sys.path insertion
    order instead of its documented fixture-discovery mechanism.
    """
    return _load_plugin_module


# ---------------------------------------------------------------------------
# Fake claude_agent_sdk message/block shapes.
#
# The plugin (like agent/claude_runtime.py) dispatches on class NAME, not
# isinstance, specifically so it behaves identically against the real
# optional extra and a stand-in. These dataclasses are usable directly by
# tests as message fixtures without needing the fake module at all.
# ---------------------------------------------------------------------------


@dataclass
class TextBlock:
    text: str


@dataclass
class ThinkingBlock:
    thinking: str
    signature: str = ""


@dataclass
class ToolUseBlock:
    id: str
    name: str
    input: dict


@dataclass
class AssistantMessage:
    content: list
    model: str = "claude-sonnet-5"
    stop_reason: str | None = None
    session_id: str | None = None
    error: str | None = None
    usage: dict | None = None


@dataclass
class ResultMessage:
    subtype: str = "success"
    session_id: str = "sdk-session-1"
    result: str | None = None
    usage: dict | None = None
    total_cost_usd: float | None = None
    terminal_reason: str | None = None
    is_error: bool = False
    errors: list | None = None


@dataclass
class StreamEvent:
    event: dict
    session_id: str = "sdk-session-1"
    uuid: str = "u1"


@dataclass
class ClaudeAgentOptions:
    system_prompt: Any = None
    setting_sources: Any = None
    strict_mcp_config: bool = False
    cwd: Any = None
    env: dict = field(default_factory=dict)
    include_partial_messages: bool = False
    allowed_tools: list = field(default_factory=list)
    tools: list = field(default_factory=list)
    max_turns: int | None = None
    mcp_servers: dict = field(default_factory=dict)
    stderr: Any = None
    model: Any = None
    effort: Any = None
    resume: Any = None


@dataclass
class HookMatcher:
    matcher: Any = None
    hooks: list = field(default_factory=list)
    timeout: Any = None


@dataclass
class ToolAnnotations:
    readOnlyHint: bool = False


def _fake_tool(name, description, input_schema, annotations=None):
    def _decorate(handler):
        return types.SimpleNamespace(
            name=name,
            description=description,
            input_schema=input_schema,
            handler=handler,
            annotations=annotations,
        )

    return _decorate


def _fake_create_sdk_mcp_server(*, name, version, tools):
    return types.SimpleNamespace(name=name, version=version, tools=list(tools))


class _FakeClaudeSDKClient:
    """Placeholder — plugin tests inject their own fakes via client_factory."""

    def __init__(self, *args, **kwargs):
        raise NotImplementedError(
            "tests must inject a client_factory/transport_factory rather than "
            "relying on the module-level ClaudeSDKClient"
        )


async def _fake_query(*args, **kwargs):
    raise NotImplementedError("fake query() is not used by the claude-sub plugin")


@pytest.fixture
def fake_sdk(monkeypatch):
    """Install a fake ``claude_agent_sdk`` module tree. Not autouse."""
    module = types.ModuleType("claude_agent_sdk")
    module.__file__ = "/fake/claude_agent_sdk/__init__.py"
    module.tool = _fake_tool
    module.create_sdk_mcp_server = _fake_create_sdk_mcp_server
    module.ToolAnnotations = ToolAnnotations
    module.ClaudeAgentOptions = ClaudeAgentOptions
    module.HookMatcher = HookMatcher
    module.ClaudeSDKClient = _FakeClaudeSDKClient
    module.query = _fake_query
    module.AssistantMessage = AssistantMessage
    module.TextBlock = TextBlock
    module.ThinkingBlock = ThinkingBlock
    module.ToolUseBlock = ToolUseBlock
    module.ResultMessage = ResultMessage
    module.StreamEvent = StreamEvent

    version_module = types.ModuleType("claude_agent_sdk._version")
    version_module.__version__ = "0.0.0-fake"

    errors_module = types.ModuleType("claude_agent_sdk._errors")

    class CLIConnectionError(Exception):
        pass

    class CLINotFoundError(Exception):
        pass

    errors_module.CLIConnectionError = CLIConnectionError
    errors_module.CLINotFoundError = CLINotFoundError

    internal_module = types.ModuleType("claude_agent_sdk._internal")

    task_compat_module = types.ModuleType("claude_agent_sdk._internal._task_compat")

    def _fake_spawn_detached(coro):
        return None

    task_compat_module.spawn_detached = _fake_spawn_detached

    transport_pkg_module = types.ModuleType("claude_agent_sdk._internal.transport")
    subprocess_cli_module = types.ModuleType(
        "claude_agent_sdk._internal.transport.subprocess_cli"
    )
    subprocess_cli_module._ACTIVE_CHILDREN = set()

    class SubprocessCLITransport:
        def __init__(self, *args, **kwargs):
            pass

    subprocess_cli_module.SubprocessCLITransport = SubprocessCLITransport

    monkeypatch.setitem(sys.modules, "claude_agent_sdk", module)
    monkeypatch.setitem(sys.modules, "claude_agent_sdk._version", version_module)
    monkeypatch.setitem(sys.modules, "claude_agent_sdk._errors", errors_module)
    monkeypatch.setitem(sys.modules, "claude_agent_sdk._internal", internal_module)
    monkeypatch.setitem(
        sys.modules, "claude_agent_sdk._internal._task_compat", task_compat_module
    )
    monkeypatch.setitem(
        sys.modules, "claude_agent_sdk._internal.transport", transport_pkg_module
    )
    monkeypatch.setitem(
        sys.modules,
        "claude_agent_sdk._internal.transport.subprocess_cli",
        subprocess_cli_module,
    )
    yield module
