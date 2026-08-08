"""Contract for the in-process ``claude_agent_sdk`` MCP bridge.

``claude-agent-sdk`` is an optional extra, so the tests stand up a minimal
fake module when it is absent — importing ``agent.transports.claude_tool_bridge``
and running this file must stay green on a machine without the SDK.
"""

import asyncio
import base64
import json
import sys
import types
from dataclasses import dataclass
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from agent.transports import claude_tool_bridge as bridge
from run_agent import AIAgent


# ---------------------------------------------------------------------------
# Fake SDK (used only when the real optional extra is not installed)
# ---------------------------------------------------------------------------


@dataclass
class _FakeToolAnnotations:
    readOnlyHint: bool = False


@dataclass
class _FakeSdkTool:
    name: str
    description: str
    input_schema: dict
    handler: object
    annotations: object = None


def _fake_tool(name, description, input_schema, annotations=None):
    def _decorate(handler):
        return _FakeSdkTool(name, description, input_schema, handler, annotations)

    return _decorate


def _fake_create_sdk_mcp_server(*, name, version, tools):
    return SimpleNamespace(name=name, version=version, tools=list(tools))


@pytest.fixture
def sdk_module(monkeypatch):
    """Yield an importable ``claude_agent_sdk``, faking it when absent."""
    try:  # pragma: no cover - exercised only where the extra is installed
        import claude_agent_sdk  # noqa: F401

        yield sys.modules["claude_agent_sdk"]
        return
    except ImportError:
        pass

    module = types.ModuleType("claude_agent_sdk")
    module.tool = _fake_tool
    module.create_sdk_mcp_server = _fake_create_sdk_mcp_server
    module.ToolAnnotations = _FakeToolAnnotations
    monkeypatch.setitem(sys.modules, "claude_agent_sdk", module)
    yield module


# ---------------------------------------------------------------------------
# Agent helpers
# ---------------------------------------------------------------------------


def _tool_defs(*names: str) -> list[dict]:
    return [
        {
            "type": "function",
            "function": {
                "name": name,
                "description": f"{name} tool",
                "parameters": {
                    "type": "object",
                    "properties": {"query": {"type": "string"}},
                    "required": ["query"],
                },
            },
        }
        for name in names
    ]


def _make_agent(*tool_names: str) -> AIAgent:
    with (
        patch("run_agent.get_tool_definitions", return_value=_tool_defs(*tool_names)),
        patch("run_agent.check_toolset_requirements", return_value={}),
        patch("hermes_cli.config.load_config", return_value={}),
        patch("hermes_cli.config.load_config_readonly", return_value={}),
        patch("run_agent.OpenAI"),
    ):
        agent = AIAgent(
            api_key="test-key-1234567890",
            base_url="https://openrouter.ai/api/v1",
            quiet_mode=True,
            skip_context_files=True,
            skip_memory=True,
        )
    agent.client = MagicMock()
    agent._use_prompt_caching = False
    agent.compression_enabled = False
    agent.save_trajectories = False
    return agent


def _tools_by_name(tools) -> dict:
    """Map tool name -> SdkMcpTool.

    ``create_sdk_mcp_server`` returns an SDK-internal value — a stub object
    carrying ``.tools`` here, a ``{"type", "name", "instance"}`` dict in the
    real package — so tests that need the tool objects go through
    ``build_bridged_sdk_tools`` and never unwrap either shape.
    """
    return {t.name: t for t in tools}


# ---------------------------------------------------------------------------
# Naming + surface
# ---------------------------------------------------------------------------


def test_module_imports_without_the_optional_sdk():
    """Nothing at module scope may require claude-agent-sdk."""
    assert "claude_agent_sdk" not in getattr(bridge, "__dict__", {})
    assert bridge.mcp_tool_name("web_search") == "mcp__hermes__web_search"


def test_missing_sdk_raises_an_actionable_import_error(monkeypatch):
    agent = _make_agent("web_search")
    # A None entry in sys.modules is the stdlib's "this import fails" marker.
    monkeypatch.setitem(sys.modules, "claude_agent_sdk", None)

    with pytest.raises(ImportError, match="claude-agent-sdk"):
        bridge.build_hermes_sdk_mcp_server(agent, "task-1")


def test_exposed_names_match_the_agents_enabled_tools_and_nothing_else(sdk_module):
    agent = _make_agent("web_search", "write_file")
    tools = bridge.build_bridged_sdk_tools(agent, "task-1")

    assert set(_tools_by_name(tools)) == {"web_search", "write_file"}
    assert set(bridge.bridged_allowed_tools(agent)) == {
        "mcp__hermes__web_search",
        "mcp__hermes__write_file",
    }
    assert all(
        name.startswith(bridge.MCP_TOOL_PREFIX)
        for name in bridge.bridged_allowed_tools(agent)
    )


def test_json_schemas_are_forwarded_verbatim(sdk_module):
    agent = _make_agent("web_search")
    tools = bridge.build_bridged_sdk_tools(agent, "task-1")

    schema = _tools_by_name(tools)["web_search"].input_schema
    assert schema == agent.tools[0]["function"]["parameters"]


def test_read_only_annotation_matches_hermes_read_only_set(sdk_module):
    from agent.tool_dispatch_helpers import _PARALLEL_SAFE_TOOLS

    read_only_name = sorted(_PARALLEL_SAFE_TOOLS)[0]
    agent = _make_agent(read_only_name, "write_file", "terminal")
    tools = bridge.build_bridged_sdk_tools(agent, "task-1")
    tools = _tools_by_name(tools)

    assert tools[read_only_name].annotations.readOnlyHint is True
    # Mutating tools must never be announced as read-only.
    assert tools["write_file"].annotations.readOnlyHint is False
    assert tools["terminal"].annotations.readOnlyHint is False


def test_read_only_predicate_never_marks_a_mutating_tool_read_only():
    for name in ("write_file", "patch", "terminal", "delegate_task", "memory"):
        assert bridge.is_read_only_tool(name) is False


# ---------------------------------------------------------------------------
# Dispatch routes through the Hermes lifecycle
# ---------------------------------------------------------------------------


def test_tool_call_reaches_execute_one_tool_and_not_handle_function_call(sdk_module):
    agent = _make_agent("web_search")
    tools = bridge.build_bridged_sdk_tools(agent, "task-routed")
    handler = _tools_by_name(tools)["web_search"].handler

    from agent.tool_executor import ToolExecutionOutcome

    seen = {}

    def _fake_execute_one_tool(bound_agent, tool_call, task_id, **kwargs):
        seen["agent"] = bound_agent
        seen["name"] = tool_call.function.name
        seen["args"] = json.loads(tool_call.function.arguments)
        seen["task_id"] = task_id
        return ToolExecutionOutcome(
            tool_call=tool_call,
            tool_call_id=tool_call.id,
            function_name=tool_call.function.name,
            function_args={},
            result=json.dumps({"ok": True}),
            duration=0.0,
            is_error=False,
            blocked=False,
            cancelled=False,
            malformed=False,
            middleware_trace=[],
        )

    with (
        patch.object(bridge, "execute_one_tool", side_effect=_fake_execute_one_tool),
        patch.object(bridge, "finalize_tool_outcome", side_effect=lambda a, o: o.result),
        patch("run_agent.handle_function_call") as raw_dispatch,
    ):
        response = asyncio.run(handler({"query": "hermes"}))

    raw_dispatch.assert_not_called()
    assert seen["agent"] is agent
    assert seen["name"] == "web_search"
    assert seen["args"] == {"query": "hermes"}
    assert seen["task_id"] == "task-routed"
    assert response["content"] == [{"type": "text", "text": json.dumps({"ok": True})}]
    assert "is_error" not in response


def test_task_id_may_be_resolved_per_call(sdk_module):
    agent = _make_agent("web_search")
    current = {"id": "task-a"}
    tools = bridge.build_bridged_sdk_tools(agent, lambda: current["id"])
    handler = _tools_by_name(tools)["web_search"].handler

    seen = []

    def _fake_dispatch(name, args, task_id, **kwargs):
        seen.append(task_id)
        return json.dumps({"ok": True})

    with (
        patch("hermes_cli.plugins.resolve_pre_tool_block", return_value=None),
        patch("run_agent.handle_function_call", side_effect=_fake_dispatch),
    ):
        asyncio.run(handler({"query": "one"}))
        current["id"] = "task-b"
        asyncio.run(handler({"query": "two"}))

    assert seen == ["task-a", "task-b"]


def test_denied_tool_call_never_dispatches_and_reports_an_error(sdk_module):
    agent = _make_agent("terminal")
    tools = bridge.build_bridged_sdk_tools(agent, "task-deny")
    handler = _tools_by_name(tools)["terminal"].handler

    with (
        patch("hermes_cli.plugins.resolve_pre_tool_block", return_value="denied by policy"),
        patch("run_agent.handle_function_call") as raw_dispatch,
    ):
        response = asyncio.run(handler({"command": "ls"}))

    raw_dispatch.assert_not_called()
    assert response["is_error"] is True
    assert "denied by policy" in response["content"][0]["text"]


# ---------------------------------------------------------------------------
# Approval context on the SDK loop thread
# ---------------------------------------------------------------------------


class _FakeAgentSession:
    """Stands in for ClaudeAgentSession so _ensure_session never spawns a CLI."""

    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.closed = False

    def ensure_started(self):
        pass


def _run_handler_like_the_sdk_loop(handler, call_args, *, count: int = 1):
    """Drive *handler* the way the SDK does: as tasks on a foreign event-loop
    thread whose context carries no gateway ContextVars.

    ``asyncio.run(handler(...))`` on the test thread inherits the test
    thread's ContextVars and masks their loss — that is exactly how the
    original suite missed the auto-approve regression. The real handler runs
    in a task spawned by the SDK's reader on the session-owned loop thread,
    where the gateway's per-turn ``copy_context()`` never propagated; an
    explicit empty ``contextvars.Context`` reproduces that.
    """
    import contextvars
    import threading

    loop = asyncio.new_event_loop()
    thread = threading.Thread(target=loop.run_forever, daemon=True)
    thread.start()
    outcomes = []
    done = threading.Event()

    def _spawn():
        tasks = [loop.create_task(handler(dict(call_args))) for _ in range(count)]

        def _finish(task):
            try:
                outcomes.append(task.result())
            except BaseException as exc:  # surfaced to the test below
                outcomes.append(exc)
            if len(outcomes) == count:
                done.set()

        for task in tasks:
            task.add_done_callback(_finish)

    loop.call_soon_threadsafe(_spawn, context=contextvars.Context())
    try:
        assert done.wait(30), "bridged handler never completed"
        for outcome in outcomes:
            if isinstance(outcome, BaseException):
                raise outcome
        return outcomes
    finally:
        loop.call_soon_threadsafe(loop.stop)
        thread.join(timeout=5)
        loop.close()


def _outcome_for(tool_call, result: str):
    from agent.tool_executor import ToolExecutionOutcome

    return ToolExecutionOutcome(
        tool_call=tool_call,
        tool_call_id=tool_call.id,
        function_name=tool_call.function.name,
        function_args={},
        result=result,
        duration=0.0,
        is_error=False,
        blocked=False,
        cancelled=False,
        malformed=False,
        middleware_trace=[],
    )


def test_gateway_approval_context_reaches_a_handler_on_the_sdk_loop_thread(
    sdk_module, monkeypatch
):
    """A dangerous command dispatched by the bridge must hit the gateway
    approval flow, never the non-interactive auto-approve branch.

    The gateway binds session identity purely via ContextVars on the Hermes
    turn thread; the SDK invokes bridge handlers on its own loop thread where
    those vars were never set. A call-time ``propagate_context_to_thread``
    there snapshots an EMPTY context, ``_is_gateway_approval_context()`` goes
    False, and ``_run_approval_gate`` auto-approves the dangerous command with
    only a log warning. The runtime must capture the turn thread's context
    (in ``_ensure_session``) and the handler must dispatch under that
    snapshot.
    """
    from agent import claude_runtime
    from agent.transports import claude_agent_session as session_mod
    from gateway.session_context import _SESSION_PLATFORM
    from tools import approval

    monkeypatch.setattr(session_mod, "ClaudeAgentSession", _FakeAgentSession)

    agent = _make_agent("terminal")
    agent._cached_system_prompt = "hermes system prompt"

    session_key = "test-bridge-approval-ctx"
    observed = {}

    def _fake_execute_one_tool(bound_agent, tool_call, task_id, **kwargs):
        # Runs on the bridge's worker thread, like the real executor stack.
        observed["is_gateway"] = approval._is_gateway_approval_context()
        observed["gate"] = approval.check_dangerous_command(
            "rm -rf /tmp/hermes-bridge-test", "local"
        )
        return _outcome_for(tool_call, json.dumps({"ok": True}))

    platform_token = _SESSION_PLATFORM.set("slack")
    key_token = approval.set_current_session_key(session_key)
    try:
        # The turn thread's half of the contract: exactly what
        # run_claude_agent_sdk_turn does before the SDK owns the turn.
        claude_runtime._ensure_session(agent, "task-approval")
        tools = bridge.build_bridged_sdk_tools(agent, "task-approval")
        handler = _tools_by_name(tools)["terminal"].handler
        with (
            patch.object(
                bridge, "execute_one_tool", side_effect=_fake_execute_one_tool
            ),
            patch.object(
                bridge, "finalize_tool_outcome", side_effect=lambda a, o: o.result
            ),
        ):
            _run_handler_like_the_sdk_loop(handler, {"command": "ls"})
    finally:
        approval.reset_current_session_key(key_token)
        _SESSION_PLATFORM.reset(platform_token)
        approval.clear_session(session_key)

    assert observed["is_gateway"] is True, (
        "bridged tool dispatch lost the gateway approval context on the SDK "
        "loop thread"
    )
    gate = observed["gate"]
    assert gate["approved"] is False, (
        f"dangerous command was auto-approved instead of gated: {gate!r}"
    )
    assert gate.get("status") == "approval_required"


def test_the_stored_context_snapshot_survives_parallel_tool_calls(
    sdk_module, monkeypatch
):
    """Claude batches read-only tools in parallel; one per-turn snapshot must
    serve concurrent workers (a shared ``contextvars.Context`` object raises
    "cannot enter context" when entered twice at once)."""
    from agent import claude_runtime
    from agent.transports import claude_agent_session as session_mod
    from gateway.session_context import _SESSION_PLATFORM
    from tools import approval

    monkeypatch.setattr(session_mod, "ClaudeAgentSession", _FakeAgentSession)

    agent = _make_agent("web_search")
    agent._cached_system_prompt = "hermes system prompt"

    import threading

    started = threading.Barrier(2, timeout=10)
    platforms = []

    def _fake_execute_one_tool(bound_agent, tool_call, task_id, **kwargs):
        started.wait()  # hold both workers inside the context at once
        platforms.append(approval._get_session_platform())
        return _outcome_for(tool_call, json.dumps({"ok": True}))

    platform_token = _SESSION_PLATFORM.set("slack")
    try:
        claude_runtime._ensure_session(agent, "task-parallel")
        tools = bridge.build_bridged_sdk_tools(agent, "task-parallel")
        handler = _tools_by_name(tools)["web_search"].handler
        with (
            patch.object(
                bridge, "execute_one_tool", side_effect=_fake_execute_one_tool
            ),
            patch.object(
                bridge, "finalize_tool_outcome", side_effect=lambda a, o: o.result
            ),
        ):
            responses = _run_handler_like_the_sdk_loop(
                handler, {"query": "hermes"}, count=2
            )
    finally:
        _SESSION_PLATFORM.reset(platform_token)

    assert len(responses) == 2
    assert all("is_error" not in r for r in responses)
    assert platforms == ["slack", "slack"]


# ---------------------------------------------------------------------------
# Result conversion
# ---------------------------------------------------------------------------


def test_error_results_carry_is_error():
    response = bridge.build_tool_response(
        "web_search", json.dumps({"error": "upstream exploded"})
    )
    assert response["is_error"] is True
    assert response["content"][0]["type"] == "text"


def test_successful_results_omit_is_error():
    response = bridge.build_tool_response("web_search", json.dumps({"results": []}))
    assert "is_error" not in response


def test_image_results_become_image_blocks_with_raw_base64():
    payload = base64.b64encode(b"\x89PNG fake").decode()
    multimodal = {
        "_multimodal": True,
        "text_summary": "a screenshot",
        "content": [
            {"type": "text", "text": "a screenshot"},
            {
                "type": "image_url",
                "image_url": {"url": f"data:image/png;base64,{payload}"},
            },
        ],
    }

    blocks = bridge.tool_result_content_blocks(multimodal)

    assert blocks[0] == {"type": "text", "text": "a screenshot"}
    image = blocks[1]
    assert image["type"] == "image"
    assert image["mimeType"] == "image/png"
    assert image["data"] == payload
    assert not image["data"].startswith("data:")
    assert base64.b64decode(image["data"]) == b"\x89PNG fake"


def test_plain_string_results_become_a_single_text_block():
    assert bridge.tool_result_content_blocks("hello") == [
        {"type": "text", "text": "hello"}
    ]


def test_tool_specs_ignore_malformed_definitions():
    agent = SimpleNamespace(
        tools=[
            {"type": "function", "function": {"name": "web_search"}},
            {"type": "function"},
            "not-a-dict",
            {"type": "function", "function": {"name": "web_search"}},
        ]
    )
    assert [spec["name"] for spec in bridge.agent_tool_specs(agent)] == ["web_search"]
