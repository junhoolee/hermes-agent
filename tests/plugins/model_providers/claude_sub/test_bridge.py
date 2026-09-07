"""``bridge.py`` — OpenAI tools -> in-process SDK MCP server + deny hook.

Uses the ``fake_sdk`` fixture (installs a fake ``claude_agent_sdk`` module
tree) since ``build_bridge``/``build_pretooluse_hooks`` import the SDK's
``tool``/``create_sdk_mcp_server``/``ToolAnnotations``/``HookMatcher`` lazily.
"""

from __future__ import annotations

import asyncio
import concurrent.futures

import pytest


TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "Read a file.",
            "parameters": {"type": "object", "properties": {"path": {"type": "string"}}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "terminal",
            "description": "Run a shell command.",
            "parameters": {"type": "object", "properties": {"command": {"type": "string"}}},
        },
    },
]


@pytest.fixture
def bridge_module(load_plugin_module):
    return load_plugin_module("bridge")


class TestBuildBridge:
    def test_two_tools_become_two_mcp_tools(self, bridge_module, fake_sdk):
        def on_call(name, args):
            fut: "concurrent.futures.Future" = concurrent.futures.Future()
            fut.set_result({"content": [{"type": "text", "text": "ok"}], "is_error": False})
            return fut

        server, allowed = bridge_module.build_bridge(TOOLS, on_call)
        assert server.name == "hermes"
        assert len(server.tools) == 2
        assert allowed == ["mcp__hermes__read_file", "mcp__hermes__terminal"]

        by_name = {t.name: t for t in server.tools}
        assert by_name["read_file"].description == "Read a file."
        assert by_name["read_file"].input_schema == TOOLS[0]["function"]["parameters"]
        assert by_name["read_file"].annotations.readOnlyHint is True
        assert by_name["terminal"].annotations.readOnlyHint is False

    def test_malformed_tool_entries_are_skipped(self, bridge_module, fake_sdk):
        def on_call(name, args):
            raise AssertionError("should not be called")

        tools = [
            {"type": "function", "function": {}},  # no name
            "not-a-dict",
            {"type": "function", "function": {"name": "  "}},  # blank name
        ]
        server, allowed = bridge_module.build_bridge(tools, on_call)
        assert server.tools == []
        assert allowed == []

    def test_handler_awaits_on_call_future_and_returns_result_verbatim(self, bridge_module, fake_sdk):
        calls = []

        def on_call(name, args):
            calls.append((name, args))
            fut: "concurrent.futures.Future" = concurrent.futures.Future()
            fut.set_result({"content": [{"type": "text", "text": "42"}], "is_error": False})
            return fut

        server, _allowed = bridge_module.build_bridge(TOOLS[:1], on_call)
        handler = server.tools[0].handler

        result = asyncio.run(handler({"path": "/tmp/x"}))
        assert result == {"content": [{"type": "text", "text": "42"}], "is_error": False}
        assert calls == [("read_file", {"path": "/tmp/x"})]

    def test_handler_propagates_error_result(self, bridge_module, fake_sdk):
        def on_call(name, args):
            fut: "concurrent.futures.Future" = concurrent.futures.Future()
            fut.set_result({"content": [{"type": "text", "text": "boom"}], "is_error": True})
            return fut

        server, _allowed = bridge_module.build_bridge(TOOLS[:1], on_call)
        handler = server.tools[0].handler
        result = asyncio.run(handler({}))
        assert result["is_error"] is True


class TestPreToolUseDenyHook:
    def _hook_fn(self, bridge_module, fake_sdk):
        hooks = bridge_module.build_pretooluse_hooks()
        matcher = hooks["PreToolUse"][0]
        assert matcher.matcher is None
        return matcher.hooks[0]

    def test_bridge_tool_call_is_allowed(self, bridge_module, fake_sdk):
        hook = self._hook_fn(bridge_module, fake_sdk)
        result = asyncio.run(hook({"tool_name": "mcp__hermes__read_file"}, "tu_1", {}))
        assert result == {}

    def test_non_bridge_tool_call_is_denied(self, bridge_module, fake_sdk):
        hook = self._hook_fn(bridge_module, fake_sdk)
        result = asyncio.run(hook({"tool_name": "Bash"}, "tu_1", {}))
        decision = result["hookSpecificOutput"]
        assert decision["hookEventName"] == "PreToolUse"
        assert decision["permissionDecision"] == "deny"
        assert "Bash" in decision["permissionDecisionReason"]
        assert "mcp__hermes__" in decision["permissionDecisionReason"]
