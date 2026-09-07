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

    def test_read_is_denied(self, bridge_module, fake_sdk):
        hook = self._hook_fn(bridge_module, fake_sdk)
        result = asyncio.run(hook({"tool_name": "Read"}, "tu_1", {}))
        assert result["hookSpecificOutput"]["permissionDecision"] == "deny"

    def test_toolsearch_is_passed_through(self, bridge_module, fake_sdk):
        hook = self._hook_fn(bridge_module, fake_sdk)
        result = asyncio.run(hook({"tool_name": "ToolSearch"}, "tu_1", {}))
        assert result == {}


class TestBuildHooks:
    """v0.1-D: PreToolUse/PostToolUse/PostToolUseFailure observation callbacks."""

    def _hooks(self, bridge_module, fake_sdk, *, on_pre=None, on_post=None):
        pre_calls: list = []
        post_calls: list = []

        def _on_pre(tool_use_id, short_name, tool_input):
            pre_calls.append((tool_use_id, short_name, tool_input))
            if on_pre is not None:
                on_pre(tool_use_id, short_name, tool_input)

        def _on_post(tool_use_id, short_name, tool_response, failed):
            post_calls.append((tool_use_id, short_name, tool_response, failed))
            if on_post is not None:
                on_post(tool_use_id, short_name, tool_response, failed)

        hooks = bridge_module.build_hooks(on_pre_tool_use=_on_pre, on_post_tool_use=_on_post)
        return hooks, pre_calls, post_calls

    def test_all_three_hook_events_are_wired(self, bridge_module, fake_sdk):
        hooks, _pre, _post = self._hooks(bridge_module, fake_sdk)
        assert set(hooks.keys()) == {"PreToolUse", "PostToolUse", "PostToolUseFailure"}
        for matchers in hooks.values():
            assert len(matchers) == 1
            assert matchers[0].matcher is None

    def test_pre_tool_use_denies_non_bridge_and_notifies_for_bridge(self, bridge_module, fake_sdk):
        hooks, pre_calls, _post = self._hooks(bridge_module, fake_sdk)
        pre_hook = hooks["PreToolUse"][0].hooks[0]

        deny = asyncio.run(pre_hook({"tool_name": "Bash"}, "tu_1", {}))
        assert deny["hookSpecificOutput"]["permissionDecision"] == "deny"
        assert pre_calls == []

        allow = asyncio.run(
            pre_hook(
                {"tool_name": "mcp__hermes__read_file", "tool_input": {"path": "/x"}}, "tu_2", {}
            )
        )
        assert allow == {}
        assert pre_calls == [("tu_2", "read_file", {"path": "/x"})]

    def test_pre_tool_use_passes_through_toolsearch_without_notifying(self, bridge_module, fake_sdk):
        hooks, pre_calls, _post = self._hooks(bridge_module, fake_sdk)
        pre_hook = hooks["PreToolUse"][0].hooks[0]
        result = asyncio.run(pre_hook({"tool_name": "ToolSearch"}, "tu_1", {}))
        assert result == {}
        assert pre_calls == []

    def test_post_tool_use_notifies_only_for_bridge_tools(self, bridge_module, fake_sdk):
        hooks, _pre, post_calls = self._hooks(bridge_module, fake_sdk)
        post_hook = hooks["PostToolUse"][0].hooks[0]

        result = asyncio.run(
            hooks["PostToolUse"][0].hooks[0](
                {"tool_name": "mcp__hermes__read_file", "tool_response": {"content": []}},
                "tu_1",
                {},
            )
        )
        assert result == {}
        assert post_calls == [("tu_1", "read_file", {"content": []}, False)]

        post_calls.clear()
        asyncio.run(post_hook({"tool_name": "ToolSearch", "tool_response": {}}, "tu_2", {}))
        assert post_calls == []

    def test_post_tool_use_failure_notifies_with_failed_true(self, bridge_module, fake_sdk):
        hooks, _pre, post_calls = self._hooks(bridge_module, fake_sdk)
        failure_hook = hooks["PostToolUseFailure"][0].hooks[0]

        asyncio.run(
            failure_hook(
                {"tool_name": "mcp__hermes__terminal", "error": "boom"}, "tu_1", {}
            )
        )
        assert post_calls == [("tu_1", "terminal", "boom", True)]

    def test_callback_exception_is_swallowed_and_hook_still_returns(self, bridge_module, fake_sdk):
        def _boom(*_args):
            raise RuntimeError("callback exploded")

        hooks, _pre, _post = self._hooks(bridge_module, fake_sdk, on_pre=_boom, on_post=_boom)
        pre_hook = hooks["PreToolUse"][0].hooks[0]
        post_hook = hooks["PostToolUse"][0].hooks[0]

        assert asyncio.run(
            pre_hook({"tool_name": "mcp__hermes__read_file", "tool_input": {}}, "tu_1", {})
        ) == {}
        assert asyncio.run(
            post_hook(
                {"tool_name": "mcp__hermes__read_file", "tool_response": {}}, "tu_1", {}
            )
        ) == {}


class TestBuildPreToolUseHooksLegacyWrapper:
    def test_wraps_build_hooks_deny_gate_only(self, bridge_module, fake_sdk):
        hooks = bridge_module.build_pretooluse_hooks()
        assert set(hooks.keys()) == {"PreToolUse"}
        hook = hooks["PreToolUse"][0].hooks[0]
        assert asyncio.run(hook({"tool_name": "mcp__hermes__read_file"}, "tu_1", {})) == {}
        deny = asyncio.run(hook({"tool_name": "Bash"}, "tu_1", {}))
        assert deny["hookSpecificOutput"]["permissionDecision"] == "deny"
