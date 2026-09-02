"""Contract for the ``claude_agent_sdk`` whole-turn runtime.

Three things are being pinned down here:

1. The turn refuses to start unless the gate is on, the optional extra is
   installed, and the user is signed in — with an error that names the fix.
2. The SDK options carry Hermes' ownership boundary: the exact Hermes system
   prompt, no SDK built-ins, no second settings load, a pinned MCP toolset.
3. A Claude turn fires the same canonical Hermes callbacks and produces the
   same message shape as every other provider, exactly once.

``claude-agent-sdk`` is an optional extra, so a stand-in module is installed
when it is absent. The SDK message/block classes are re-declared locally
because the runtime dispatches on class name — that is what lets the projector
behave identically with and without the real package.
"""

from __future__ import annotations

import asyncio
import sys
import types
from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from agent import claude_runtime
from agent.claude_runtime import (
    ClaudeEventProjector,
    build_claude_agent_options,
    claude_runtime_preflight,
    display_tool_name,
    run_claude_agent_sdk_turn,
)
from agent.transports.claude_tool_bridge import MCP_SERVER_NAME, bridged_allowed_tools
from run_agent import AIAgent


# ---------------------------------------------------------------------------
# SDK stand-in (used only when the optional extra is not installed)
# ---------------------------------------------------------------------------


@dataclass
class _FakeToolAnnotations:
    readOnlyHint: bool = False


@dataclass
class _FakeSdkTool:
    name: str
    description: str
    input_schema: dict
    handler: Any
    annotations: Any = None


def _fake_tool(name, description, input_schema, annotations=None):
    def _decorate(handler):
        return _FakeSdkTool(name, description, input_schema, handler, annotations)

    return _decorate


def _fake_create_sdk_mcp_server(*, name, version, tools):
    return SimpleNamespace(name=name, version=version, tools=list(tools))


@dataclass
class _FakeClaudeAgentOptions:
    system_prompt: Any = None
    tools: Any = None
    allowed_tools: list = field(default_factory=list)
    disallowed_tools: list = field(default_factory=list)
    mcp_servers: dict = field(default_factory=dict)
    strict_mcp_config: bool = False
    setting_sources: Any = None
    cwd: Any = None
    env: dict = field(default_factory=dict)
    stderr: Any = None
    model: Any = None
    include_partial_messages: bool = False
    resume: Any = None
    session_store: Any = None
    hooks: Any = None


@dataclass
class _FakeHookMatcher:
    matcher: Any = None
    hooks: list = field(default_factory=list)
    timeout: Any = None


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
    module.ClaudeAgentOptions = _FakeClaudeAgentOptions
    module.HookMatcher = _FakeHookMatcher
    monkeypatch.setitem(sys.modules, "claude_agent_sdk", module)
    yield module


# ---------------------------------------------------------------------------
# SDK message shapes (dispatch is by class name)
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
class ToolResultBlock:
    tool_use_id: str
    content: Any = None
    is_error: bool | None = None


@dataclass
class AssistantMessage:
    content: list
    model: str = "claude-sonnet-4-5"
    stop_reason: str | None = None
    session_id: str | None = None
    error: str | None = None
    # Per-call API usage, mirroring the real SDK class (types.py) — each
    # AssistantMessage carries the usage of the single API call it came from.
    usage: dict | None = None


@dataclass
class UserMessage:
    content: Any


@dataclass
class SystemMessage:
    subtype: str
    data: dict = field(default_factory=dict)


@dataclass
class StreamEvent:
    event: dict
    session_id: str = "sdk-session-1"
    uuid: str = "u1"


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


def _text_delta(text: str) -> StreamEvent:
    return StreamEvent(
        event={"type": "content_block_delta", "delta": {"type": "text_delta", "text": text}}
    )


def _thinking_delta(text: str) -> StreamEvent:
    return StreamEvent(
        event={
            "type": "content_block_delta",
            "delta": {"type": "thinking_delta", "thinking": text},
        }
    )


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
                },
            },
        }
        for name in names
    ]


def _make_agent(*tool_names: str, api_mode: str = "claude_agent_sdk") -> AIAgent:
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
    agent.api_mode = api_mode
    agent.client = MagicMock()
    agent._use_prompt_caching = False
    agent.compression_enabled = False
    agent.save_trajectories = False
    return agent


class _Recorder:
    """Captures the canonical Hermes callbacks a runtime must fire."""

    def __init__(self, agent) -> None:
        self.text: list[str] = []
        self.reasoning: list[str] = []
        self.interim: list[dict] = []
        self.progress: list[tuple] = []
        self.tool_started: list[tuple] = []
        self.tool_completed: list[tuple] = []
        agent._fire_stream_delta = self.text.append
        agent._fire_reasoning_delta = self.reasoning.append
        agent._emit_interim_assistant_message = self.interim.append
        agent.tool_progress_callback = self._progress
        agent.tool_start_callback = lambda *a: self.tool_started.append(a)
        agent.tool_complete_callback = lambda *a: self.tool_completed.append(a)

    def _progress(self, phase, name, *rest, **kwargs):
        self.progress.append((phase, name, kwargs))


# ---------------------------------------------------------------------------
# Preflight
# ---------------------------------------------------------------------------


def test_preflight_names_the_config_key_when_the_gate_is_off():
    message = claude_runtime_preflight({"claude_subscription": {"enabled": False}})
    assert message is not None
    assert "claude_subscription.enabled" in message


def test_preflight_names_the_extra_when_the_sdk_is_missing():
    with patch(
        "hermes_cli.claude_subscription.claude_agent_sdk_available", return_value=False
    ):
        message = claude_runtime_preflight({"claude_subscription": {"enabled": True}})
    assert message is not None
    assert "claude-code" in message


def test_preflight_names_the_login_command_when_signed_out():
    with (
        patch(
            "hermes_cli.claude_subscription.claude_agent_sdk_available",
            return_value=True,
        ),
        patch(
            "hermes_cli.claude_code.probe_claude_auth_cached",
            return_value={"logged_in": False, "message": ""},
        ),
    ):
        message = claude_runtime_preflight({"claude_subscription": {"enabled": True}})
    assert message is not None
    assert "claude auth login" in message


def test_failure_result_attaches_leftover_pending_steer():
    """A /steer sent right before a refused/failed turn (preflight gate,
    billing refusal, session-construction failure) must still surface —
    _failure_result is the early-return path run_claude_agent_sdk_turn
    takes before any bridged tool call could have delivered it."""
    agent = _make_agent("web_search")
    agent._pending_steer = "wait, use the other branch"

    result = claude_runtime._failure_result(
        agent, [], final_response="refused", error="refused"
    )

    assert result["pending_steer"] == "wait, use the other branch"
    assert agent._pending_steer is None


def test_failure_result_omits_pending_steer_key_when_none_pending():
    agent = _make_agent("web_search")

    result = claude_runtime._failure_result(
        agent, [], final_response="refused", error="refused"
    )

    assert "pending_steer" not in result


def test_preflight_passes_when_all_three_gates_are_open():
    with (
        patch(
            "hermes_cli.claude_subscription.claude_agent_sdk_available",
            return_value=True,
        ),
        patch(
            "hermes_cli.claude_code.probe_claude_auth_cached",
            return_value={"logged_in": True, "message": "Signed in."},
        ),
    ):
        assert claude_runtime_preflight({"claude_subscription": {"enabled": True}}) is None


@pytest.mark.parametrize(
    "config,patches",
    [
        ({"claude_subscription": {"enabled": False}}, {}),
        ({"claude_subscription": {"enabled": True}}, {"sdk": False}),
        ({"claude_subscription": {"enabled": True}}, {"sdk": True, "login": False}),
    ],
)
def test_a_refused_turn_never_builds_a_session(config, patches):
    agent = _make_agent("web_search")
    messages: list[dict] = []
    stack = [patch("hermes_cli.config.load_config_readonly", return_value=config)]
    if "sdk" in patches:
        stack.append(
            patch(
                "hermes_cli.claude_subscription.claude_agent_sdk_available",
                return_value=patches["sdk"],
            )
        )
    if "login" in patches:
        stack.append(
            patch(
                "hermes_cli.claude_code.probe_claude_auth_cached",
                return_value={"logged_in": patches["login"], "message": ""},
            )
        )
    built = []
    stack.append(
        patch.object(
            claude_runtime, "_ensure_session", lambda *a, **k: built.append(a)
        )
    )

    from contextlib import ExitStack

    with ExitStack() as es:
        for ctx in stack:
            es.enter_context(ctx)
        result = run_claude_agent_sdk_turn(
            agent,
            user_message="hi",
            original_user_message="hi",
            messages=messages,
            effective_task_id="task-1",
        )

    assert built == []
    assert result["completed"] is False
    assert result["failed"] is True
    assert result["error"]
    assert messages == []


# ---------------------------------------------------------------------------
# Options — the ownership boundary
# ---------------------------------------------------------------------------


def _options_for(agent, prompt="HERMES SYSTEM PROMPT"):
    return build_claude_agent_options(
        agent,
        system_prompt=prompt,
        effective_task_id=lambda: "task-1",
        cwd="/tmp/workspace",
    )


def test_options_append_hermes_own_identity_and_author_nothing(sdk_module):
    from agent.prompt_builder import DEFAULT_AGENT_IDENTITY

    agent = _make_agent("web_search")
    agent._cached_system_prompt = "SYSTEM PROMPT — byte stable"
    options = _options_for(agent, agent._cached_system_prompt)

    # The append is Hermes' own identity section verbatim — no text authored
    # for this provider. The classifier bills a preset-replacing request, or
    # one carrying the full prompt, to extra usage (decision record §11), so
    # the rest of the prompt rides the first user turn instead.
    assert options.system_prompt == {
        "type": "preset",
        "preset": "claude_code",
        "append": claude_runtime.claude_subscription_append(agent),
    }
    append = options.system_prompt["append"]
    assert append.startswith(DEFAULT_AGENT_IDENTITY.strip())
    # The only non-Hermes text is the factual tool-routing note — no persona.
    assert claude_runtime.CLAUDE_TOOL_ROUTING_NOTE in append
    assert "mcp__hermes__" in append
    # The full prompt must NOT be in the system slot.
    assert agent._cached_system_prompt not in str(options.system_prompt)


def test_identity_anchor_follows_a_customised_soul(sdk_module, monkeypatch):
    # A user who customises SOUL.md gets their customisation in the anchor
    # too, rather than a provider-specific persona overriding it.
    monkeypatch.setattr(
        "agent.prompt_builder.load_soul_md", lambda *a, **k: "You are Ares Agent."
    )
    assert claude_runtime.hermes_identity_anchor() == "You are Ares Agent."


def test_options_keep_builtins_in_context_but_deny_them_via_hook(sdk_module):
    agent = _make_agent("web_search", "terminal")
    options = _options_for(agent)

    # Built-ins stay context-visible (stripping them trips the billing
    # classifier); execution is pinned to the bridge by the PreToolUse hook.
    assert getattr(options, "tools", None) in (None, ()) or options.tools is None
    hooks = options.hooks or {}
    assert "PreToolUse" in hooks

    matcher = hooks["PreToolUse"][0]
    hook = matcher.hooks[0]

    async def _run(name):
        return await hook({"tool_name": name}, "toolu_1", None)

    denied = asyncio.run(_run("Bash"))
    decision = denied["hookSpecificOutput"]
    assert decision["permissionDecision"] == "deny"
    assert "mcp__hermes__" in decision["permissionDecisionReason"]
    # Read is auto-allowed by the CLI without a permission prompt, so the hook
    # (not can_use_tool) must be the choke point for it too.
    assert asyncio.run(_run("Read"))["hookSpecificOutput"]["permissionDecision"] == "deny"
    assert asyncio.run(_run("mcp__hermes__web_search")) == {}


def test_options_expose_only_the_hermes_bridge(sdk_module):
    agent = _make_agent("web_search", "terminal")
    options = _options_for(agent)

    assert set(options.mcp_servers) == {MCP_SERVER_NAME}
    assert sorted(options.allowed_tools) == sorted(bridged_allowed_tools(agent))
    assert all(name.startswith("mcp__hermes__") for name in options.allowed_tools)
    # Every allowed name is actually served by the bridge.
    assert len(options.allowed_tools) == len(agent.tools)


def test_options_pin_the_toolset_and_skip_a_second_settings_load(sdk_module):
    agent = _make_agent("web_search")
    options = _options_for(agent)

    assert options.setting_sources == []
    assert options.strict_mcp_config is True


def test_options_carry_cwd_and_model_and_hold_no_credential(sdk_module):
    agent = _make_agent("web_search")
    agent.model = "claude-sonnet-4-5"
    options = _options_for(agent)

    assert options.cwd == "/tmp/workspace"
    assert options.model == "claude-sonnet-4-5"
    # PR7 owns the sanitized environment; PR4 must not smuggle credentials in.
    assert options.env == {}


def test_options_leave_the_pr5_resume_seam_unset(sdk_module):
    agent = _make_agent("web_search")
    options = _options_for(agent)

    assert getattr(options, "resume", None) is None
    assert getattr(options, "session_store", None) is None


# ---------------------------------------------------------------------------
# Event projection
# ---------------------------------------------------------------------------


def test_mcp_namespacing_is_stripped_for_display():
    assert display_tool_name("mcp__hermes__web_search") == "web_search"
    assert display_tool_name("Bash") == "Bash"


def test_a_multi_tool_streaming_turn_fires_the_canonical_callbacks():
    agent = _make_agent("web_search", "terminal")
    rec = _Recorder(agent)
    projector = ClaudeEventProjector(agent)

    for message in [
        SystemMessage("init", {"session_id": "sdk-session-1"}),
        _thinking_delta("planning"),
        _text_delta("Looking that up"),
        AssistantMessage(
            content=[
                ThinkingBlock("planning"),
                TextBlock("Looking that up"),
                ToolUseBlock("t1", "mcp__hermes__web_search", {"query": "hermes"}),
                ToolUseBlock("t2", "mcp__hermes__terminal", {"command": "ls"}),
            ],
            session_id="sdk-session-1",
        ),
        UserMessage(content=[ToolResultBlock("t1", "search results")]),
        UserMessage(content=[ToolResultBlock("t2", "file-a\nfile-b")]),
        _text_delta("All done"),
        AssistantMessage(content=[TextBlock("All done")]),
        ResultMessage(result="All done", usage={"input_tokens": 10, "output_tokens": 4}),
    ]:
        projector(message)
    projector.finalize()

    assert rec.text == ["Looking that up", "All done"]
    assert rec.reasoning == ["planning"]
    started = [p for p in rec.progress if p[0] == "tool.started"]
    completed = [p for p in rec.progress if p[0] == "tool.completed"]
    assert [p[1] for p in started] == ["web_search", "terminal"]
    assert [p[1] for p in completed] == ["web_search", "terminal"]
    assert all(p[2]["is_error"] is False for p in completed)
    assert [c[1] for c in rec.tool_started] == ["web_search", "terminal"]
    assert [c[1] for c in rec.tool_completed] == ["web_search", "terminal"]
    assert projector.tool_iterations == 2


def test_every_dispatched_message_touches_the_activity_heartbeat(monkeypatch):
    """A quiet stretch between SDK messages must not look like a stalled
    turn to the generic watchdogs that key off ``_last_activity_ts``
    (kanban worker reclaim, delegate subagent timeout diagnostics)."""
    agent = _make_agent("web_search")
    _Recorder(agent)
    touches: list[str] = []
    agent._touch_activity = lambda desc: touches.append(desc)
    projector = ClaudeEventProjector(agent)

    # Space dispatches beyond the throttle window so each message lands as
    # its own heartbeat tick, matching a real turn where messages are
    # seconds apart rather than back-to-back in a test loop.
    clock = [0.0]

    def _fake_monotonic() -> float:
        clock[0] += claude_runtime._STREAM_HEARTBEAT_THROTTLE_SECONDS + 0.1
        return clock[0]

    monkeypatch.setattr(claude_runtime.time, "monotonic", _fake_monotonic)

    messages = [
        SystemMessage("init", {"session_id": "sdk-session-1"}),
        _text_delta("Looking that up"),
        AssistantMessage(content=[TextBlock("Looking that up")]),
        ResultMessage(result="Looking that up"),
    ]
    for message in messages:
        projector(message)

    assert len(touches) == len(messages)
    assert all("claude sdk" in desc for desc in touches)


def test_rapid_stream_events_throttle_to_a_single_heartbeat_touch():
    """``StreamEvent`` fires once per content-block delta — roughly once per
    token during a streamed turn. Touching ``_last_activity_ts`` on every
    single one is pure overhead for watchdogs that only care about
    staleness on the order of seconds to minutes (see
    ``_STREAM_HEARTBEAT_THROTTLE_SECONDS``), so a burst arriving within one
    wall-clock instant must collapse to a single touch."""
    agent = _make_agent("web_search")
    _Recorder(agent)
    touches: list[str] = []
    agent._touch_activity = lambda desc: touches.append(desc)
    projector = ClaudeEventProjector(agent)

    for _ in range(25):
        projector(_text_delta("a"))

    assert len(touches) == 1


def test_dispatch_survives_an_agent_stand_in_without_touch_activity():
    """``self._agent`` is sometimes a lightweight stand-in that doesn't
    define every ``AIAgent`` method. The heartbeat touch must go through
    the same ``getattr`` guard the display callbacks already get via
    ``_fire`` — a missing ``_touch_activity`` must never raise out of
    dispatch."""
    agent = SimpleNamespace()
    recorder = _Recorder(agent)
    assert not hasattr(agent, "_touch_activity")
    projector = ClaudeEventProjector(agent)

    projector(SystemMessage("init", {"session_id": "sdk-session-1"}))
    projector(_text_delta("hi"))
    projector(AssistantMessage(content=[TextBlock("hi")]))
    projector(ResultMessage(result="hi"))

    assert recorder.text == ["hi"]


def test_streamed_text_is_not_replayed_when_the_block_completes():
    agent = _make_agent("web_search")
    rec = _Recorder(agent)
    projector = ClaudeEventProjector(agent)

    projector(_text_delta("hel"))
    projector(_text_delta("lo"))
    projector(AssistantMessage(content=[TextBlock("hello")]))
    projector(ResultMessage(result="hello"))

    assert rec.text == ["hel", "lo"]


def test_completed_blocks_still_stream_when_partial_events_are_absent():
    agent = _make_agent("web_search")
    rec = _Recorder(agent)
    projector = ClaudeEventProjector(agent)

    projector(AssistantMessage(content=[TextBlock("hello")]))
    projector(ResultMessage(result="hello"))

    assert rec.text == ["hello"]


def test_projected_messages_preserve_role_alternation():
    agent = _make_agent("web_search", "terminal")
    projector = ClaudeEventProjector(agent)

    for message in [
        AssistantMessage(
            content=[
                TextBlock("working"),
                ToolUseBlock("t1", "mcp__hermes__web_search", {"query": "a"}),
                ToolUseBlock("t2", "mcp__hermes__terminal", {"command": "ls"}),
            ]
        ),
        # Results land out of order; the transcript must still answer the
        # tool_calls in the order they were issued.
        UserMessage(content=[ToolResultBlock("t2", "listing")]),
        UserMessage(content=[ToolResultBlock("t1", "results")]),
        AssistantMessage(content=[TextBlock("done")]),
        ResultMessage(result="done"),
    ]:
        projector(message)
    projected = projector.finalize()

    roles = [m["role"] for m in projected]
    assert roles == ["assistant", "tool", "tool", "assistant"]
    assert [m["tool_call_id"] for m in projected if m["role"] == "tool"] == ["t1", "t2"]
    call_ids = [tc["id"] for tc in projected[0]["tool_calls"]]
    assert call_ids == ["t1", "t2"]
    assert [tc["function"]["name"] for tc in projected[0]["tool_calls"]] == [
        "web_search",
        "terminal",
    ]
    # Never two assistant messages in a row.
    assert not any(
        roles[i] == roles[i + 1] == "assistant" for i in range(len(roles) - 1)
    )


def test_consecutive_assistant_text_is_merged_rather_than_duplicated():
    agent = _make_agent("web_search")
    projector = ClaudeEventProjector(agent)

    projector(AssistantMessage(content=[TextBlock("part one")]))
    projector(AssistantMessage(content=[TextBlock("part two")]))
    projected = projector.finalize()

    assert [m["role"] for m in projected] == ["assistant"]
    assert "part one" in projected[0]["content"]
    assert "part two" in projected[0]["content"]


def test_an_unanswered_tool_call_still_gets_a_tool_message():
    """An unanswered tool_call_id makes the next provider request fail."""
    agent = _make_agent("web_search")
    projector = ClaudeEventProjector(agent)

    projector(
        AssistantMessage(
            content=[ToolUseBlock("t1", "mcp__hermes__web_search", {"query": "a"})]
        )
    )
    projector(ResultMessage(result="gave up"))
    projected = projector.finalize()

    assert [m["role"] for m in projected] == ["assistant", "tool"]
    assert projected[1]["tool_call_id"] == "t1"
    assert projected[1]["content"]


def test_the_sdk_session_id_is_captured_for_pr5():
    agent = _make_agent("web_search")
    projector = ClaudeEventProjector(agent)

    projector(SystemMessage("init", {"session_id": "sdk-session-xyz"}))
    projector(ResultMessage(session_id="sdk-session-xyz"))

    assert projector.session_id == "sdk-session-xyz"
    assert agent._claude_sdk_session_id == "sdk-session-xyz"


def test_a_compaction_boundary_is_observed():
    agent = _make_agent("web_search")
    projector = ClaudeEventProjector(agent)

    projector(SystemMessage("compact_boundary", {"session_id": "s"}))
    assert projector.compacted is True


def test_terminal_reason_and_error_are_surfaced():
    agent = _make_agent("web_search")
    projector = ClaudeEventProjector(agent)

    projector(
        ResultMessage(
            subtype="error_during_execution",
            is_error=True,
            errors=["upstream refused"],
            terminal_reason="max_turns",
        )
    )

    assert projector.is_error is True
    assert "upstream refused" in projector.error
    assert projector.terminal_reason == "max_turns"


def test_image_tool_results_survive_projection():
    agent = _make_agent("web_search")
    agent._model_supports_vision = lambda *a, **k: True
    agent._provider_supports_vision_tool_messages = lambda *a, **k: True
    projector = ClaudeEventProjector(agent)

    projector(
        AssistantMessage(
            content=[ToolUseBlock("t1", "mcp__hermes__web_search", {"query": "a"})]
        )
    )
    projector(
        UserMessage(
            content=[
                ToolResultBlock(
                    "t1",
                    [
                        {"type": "text", "text": "a screenshot"},
                        {"type": "image", "data": "QUJD", "mimeType": "image/png"},
                    ],
                )
            ]
        )
    )
    projected = projector.finalize()

    content = projected[1]["content"]
    assert isinstance(content, list)
    assert any(part.get("type") == "image_url" for part in content)


def test_a_buggy_display_callback_cannot_break_the_turn():
    agent = _make_agent("web_search")

    def _boom(*args, **kwargs):
        raise RuntimeError("display exploded")

    agent._fire_stream_delta = _boom
    agent.tool_progress_callback = _boom
    projector = ClaudeEventProjector(agent)

    projector(AssistantMessage(content=[TextBlock("hello")]))
    projector(ResultMessage(result="hello"))

    assert projector.final_text == "hello"


# ---------------------------------------------------------------------------
# Whole turn
# ---------------------------------------------------------------------------


class _StubSession:
    """Replays a scripted message list through the projector."""

    def __init__(self, script, *, raises=None):
        self.script = script
        self.raises = raises
        self.closed = False
        self.prompts = []
        self.session_ids = []
        self.last_stall_timeout = "unset"
        self.last_stall_exempt = None
        self.interrupt_requests = 0
        self.nowait_interrupt_requests = 0

    def run_turn(
        self, prompt, *, on_message, timeout=None, stall_timeout=None, stall_exempt=None
    ):
        self.prompts.append(prompt)
        self.last_stall_timeout = stall_timeout
        self.last_stall_exempt = stall_exempt
        if self.raises is not None:
            raise self.raises
        for message in self.script:
            on_message(message)
        return len(self.script)

    def note_session_id(self, session_id):
        self.session_ids.append(session_id)

    def request_interrupt(self):
        self.interrupt_requests += 1
        return True

    def request_interrupt_nowait(self):
        self.nowait_interrupt_requests += 1
        return True

    def close(self):
        self.closed = True


def _run_turn(agent, session, messages=None):
    messages = messages if messages is not None else []
    with (
        patch.object(claude_runtime, "claude_runtime_preflight", return_value=None),
        # The per-session billing proof spawns the real CLI; a turn-shape test
        # must not depend on the developer's (or CI's) Claude login.
        patch.object(claude_runtime, "verify_claude_billing_for_agent", return_value=None),
        patch.object(claude_runtime, "_ensure_session", return_value=session),
    ):
        result = run_claude_agent_sdk_turn(
            agent,
            user_message="hi",
            original_user_message="hi",
            messages=messages,
            effective_task_id="task-1",
        )
    return result, messages


def test_a_completed_turn_returns_the_run_conversation_shape():
    agent = _make_agent("web_search")
    session = _StubSession(
        [
            AssistantMessage(content=[TextBlock("hello")]),
            ResultMessage(
                result="hello",
                usage={"input_tokens": 12, "output_tokens": 3},
                total_cost_usd=0.001,
            ),
        ]
    )
    result, messages = _run_turn(agent, session)

    assert result["final_response"] == "hello"
    assert result["messages"] is messages
    assert result["completed"] is True
    assert result["partial"] is False
    assert result["interrupted"] is False
    assert result["error"] is None
    assert result["api_calls"] == 1
    assert result["agent_persisted"] is True
    assert result["claude_session_id"] == "sdk-session-1"
    assert result["prompt_tokens"] >= 12


def test_leftover_pending_steer_is_attached_to_the_completed_turn_result():
    """A /steer that arrives after the last bridged tool call (or with no
    tool calls at all) must not be silently dropped — it rides back on the
    turn result so the caller can requeue it as the next user turn, the same
    contract chat_completions' turn_finalizer already honors."""
    agent = _make_agent("web_search")
    agent._pending_steer = "actually check the tests too"
    session = _StubSession([AssistantMessage(content=[TextBlock("hello")]), ResultMessage(result="hello")])

    result, _messages = _run_turn(agent, session)

    assert result["pending_steer"] == "actually check the tests too"
    assert agent._pending_steer is None


def test_a_completed_turn_with_no_pending_steer_omits_the_key():
    agent = _make_agent("web_search")
    session = _StubSession([AssistantMessage(content=[TextBlock("hello")]), ResultMessage(result="hello")])

    result, _messages = _run_turn(agent, session)

    assert "pending_steer" not in result


def test_projected_messages_are_spliced_exactly_once_including_trailing_events():
    """Events after ResultMessage must be projected, and only once."""
    agent = _make_agent("web_search")
    session = _StubSession(
        [
            AssistantMessage(
                content=[ToolUseBlock("t1", "mcp__hermes__web_search", {"query": "a"})]
            ),
            UserMessage(content=[ToolResultBlock("t1", "results")]),
            ResultMessage(result="done"),
            # Trailing frame the CLI flushed after the result.
            AssistantMessage(content=[TextBlock("done")]),
        ]
    )
    result, messages = _run_turn(agent, session)

    assert [m["role"] for m in messages] == ["assistant", "tool", "assistant"]
    assert sum(1 for m in messages if m["role"] == "tool") == 1
    assert messages[-1]["content"] == "done"
    # A second finalize must not be able to duplicate anything.
    assert result["messages"] == messages


def test_a_wedged_turn_retires_the_session_so_the_next_one_respawns():
    agent = _make_agent("web_search")
    agent._claude_session = session = _StubSession([], raises=TimeoutError("stalled"))
    result, _messages = _run_turn(agent, session)

    assert session.closed is True
    assert getattr(agent, "_claude_session", None) is None
    assert result["completed"] is False
    assert "stalled" in result["error"]


def test_an_is_error_result_hands_the_turn_off_as_a_failed_attempt():
    """A CLI-reported error (e.g. a session-limit ResultMessage) with no
    exception, no cap, and no user interrupt must surface as a real turn
    failure — ``failed: True`` — so the caller's fallback chain (Codex etc.)
    actually activates, instead of silently returning the error text as a
    normal ``completed: False`` assistant reply."""
    agent = _make_agent("web_search")
    agent._claude_session = session = _StubSession(
        [
            ResultMessage(
                subtype="success",
                is_error=True,
                result="You've hit your session limit · resets 6pm (Asia/Seoul)",
            )
        ]
    )
    result, messages = _run_turn(agent, session)

    assert result["failed"] is True
    assert result["completed"] is False
    assert "session limit" in result["error"]
    assert "session limit" in result["final_response"]
    # The failed attempt's assistant output must not land in the transcript
    # — the fallback provider re-processes the turn from the trailing user
    # message, so an appended assistant row would break role alternation.
    assert messages == []
    # The wedged SDK session must not be reused by the next (fallback) turn.
    assert session.closed is True
    assert getattr(agent, "_claude_session", None) is None


def test_tool_iterations_feed_the_skill_nudge_counter():
    agent = _make_agent("web_search")
    agent._iters_since_skill = 0
    session = _StubSession(
        [
            AssistantMessage(
                content=[ToolUseBlock("t1", "mcp__hermes__web_search", {"query": "a"})]
            ),
            UserMessage(content=[ToolResultBlock("t1", "results")]),
            ResultMessage(result="done"),
        ]
    )
    _run_turn(agent, session)

    assert agent._iters_since_skill == 1


def test_a_turn_without_usage_still_counts_as_one_api_call():
    agent = _make_agent("web_search")
    before = agent.session_api_calls
    session = _StubSession([ResultMessage(result="hi")])
    result, _messages = _run_turn(agent, session)

    assert agent.session_api_calls == before + 1
    assert result["api_calls"] == 1


# ---------------------------------------------------------------------------
# Every other api_mode is untouched
# ---------------------------------------------------------------------------


def _exhaust_budget(agent):
    agent.max_iterations = 0
    agent.iteration_budget._used = agent.iteration_budget.max_total


@pytest.mark.parametrize(
    "api_mode",
    [
        "chat_completions",
        "anthropic_messages",
        "codex_responses",
        "codex_app_server",
        "bedrock_converse",
    ],
)
def test_the_early_branch_does_not_fire_for_other_api_modes(api_mode):
    agent = _make_agent("web_search", api_mode=api_mode)
    _exhaust_budget(agent)
    fired = []
    agent._run_claude_agent_sdk_turn = lambda **kw: fired.append(kw) or {}
    agent._run_codex_app_server_turn = lambda **kw: {
        "final_response": "codex",
        "messages": [],
        "api_calls": 0,
    }

    agent.run_conversation("hello")

    assert fired == []


def test_the_early_branch_fires_for_claude_agent_sdk():
    agent = _make_agent("web_search", api_mode="claude_agent_sdk")
    _exhaust_budget(agent)
    fired = []

    def _forward(**kwargs):
        fired.append(kwargs)
        return {"final_response": "claude", "messages": [], "api_calls": 1}

    agent._run_claude_agent_sdk_turn = _forward
    result = agent.run_conversation("hello")

    assert len(fired) == 1
    assert result["final_response"] == "claude"
    # The default provider loop was bypassed entirely.
    assert result["api_calls"] == 1


# ---------------------------------------------------------------------------
# Session start timeout is configurable
# ---------------------------------------------------------------------------


class _CapturingSession:
    """Stands in for ClaudeAgentSession; records constructor kwargs."""

    last_kwargs: dict = {}

    def __init__(self, **kwargs):
        type(self).last_kwargs = kwargs
        self.closed = False

    def ensure_started(self):
        pass


def _ensure_session_with_config(monkeypatch, config: dict):
    from agent.transports import claude_agent_session as session_mod

    monkeypatch.setattr(session_mod, "ClaudeAgentSession", _CapturingSession)
    _CapturingSession.last_kwargs = {}
    agent = _make_agent("web_search")
    agent._cached_system_prompt = "hermes system prompt"
    with patch("hermes_cli.config.load_config_readonly", return_value=config):
        claude_runtime._ensure_session(agent, "task-timeout")
    return _CapturingSession.last_kwargs


def test_start_timeout_follows_claude_subscription_config(monkeypatch):
    """`claude_subscription.start_timeout` must reach the session — a 60s
    hardcode is what let a cold resume of a large session on slow hardware
    blow the startup deadline (2026-08-09 stall incident)."""
    kwargs = _ensure_session_with_config(
        monkeypatch,
        {"claude_subscription": {"enabled": True, "start_timeout": 120}},
    )
    assert kwargs.get("start_timeout") == 120.0


def test_start_timeout_defaults_to_60_seconds_when_not_configured(monkeypatch):
    kwargs = _ensure_session_with_config(
        monkeypatch, {"claude_subscription": {"enabled": True}}
    )
    # Omitted, so ClaudeAgentSession's own default applies.
    assert "start_timeout" not in kwargs

    from agent.transports.claude_agent_session import DEFAULT_START_TIMEOUT_SECONDS

    assert DEFAULT_START_TIMEOUT_SECONDS == 60.0


def test_start_timeout_ignores_a_malformed_config_value(monkeypatch):
    kwargs = _ensure_session_with_config(
        monkeypatch,
        {"claude_subscription": {"enabled": True, "start_timeout": "soon"}},
    )
    assert "start_timeout" not in kwargs


# ---------------------------------------------------------------------------
# Stall watchdog wiring
# ---------------------------------------------------------------------------


def test_stall_timeout_defaults_to_300_seconds_when_not_configured():
    with patch(
        "hermes_cli.config.load_config_readonly",
        return_value={"claude_subscription": {"enabled": True}},
    ):
        assert claude_runtime._configured_stall_timeout() == 300.0


def test_stall_timeout_follows_claude_subscription_config():
    with patch(
        "hermes_cli.config.load_config_readonly",
        return_value={"claude_subscription": {"enabled": True, "stall_timeout": 90}},
    ):
        assert claude_runtime._configured_stall_timeout() == 90.0


def test_stall_timeout_zero_disables_the_watchdog():
    with patch(
        "hermes_cli.config.load_config_readonly",
        return_value={"claude_subscription": {"enabled": True, "stall_timeout": 0}},
    ):
        assert claude_runtime._configured_stall_timeout() is None


def test_stall_timeout_ignores_a_malformed_config_value():
    with patch(
        "hermes_cli.config.load_config_readonly",
        return_value={"claude_subscription": {"enabled": True, "stall_timeout": "soon"}},
    ):
        assert claude_runtime._configured_stall_timeout() == 300.0


def test_stall_timeout_falls_back_to_the_default_when_config_load_fails():
    with patch(
        "hermes_cli.config.load_config_readonly", side_effect=RuntimeError("boom")
    ):
        assert claude_runtime._configured_stall_timeout() == 300.0


def test_run_turn_receives_the_configured_stall_timeout_and_bridge_exempt():
    agent = _make_agent("web_search")
    session = _StubSession([ResultMessage(result="hi")])
    with patch(
        "hermes_cli.config.load_config_readonly",
        return_value={"claude_subscription": {"enabled": True, "stall_timeout": 45}},
    ):
        _run_turn(agent, session)

    assert session.last_stall_timeout == 45.0
    assert callable(session.last_stall_exempt)
    assert session.last_stall_exempt() is False
    agent._claude_bridge_inflight = 1
    assert session.last_stall_exempt() is True


# ---------------------------------------------------------------------------
# last_prompt_tokens approximates the LIVE context, not the turn total
# ---------------------------------------------------------------------------


def test_last_prompt_tokens_is_the_last_calls_context_not_the_turn_sum():
    """``ResultMessage.usage`` is turn-CUMULATIVE: every internal API call's
    input/cache tokens are summed, so a multi-tool turn reports a "prompt"
    several times larger than the live context (observed 2026-08-09: a
    ~26K-token transcript reported 1,079,680 "actual" tokens and tripped
    gateway hygiene).  Live-context consumers (hygiene, compression
    thresholds, the context footer) must get the LAST call's context size —
    input + cache_read + cache_write of the final API call — while billing
    keeps the cumulative figures."""
    agent = _make_agent("web_search")
    prompt_before = agent.session_prompt_tokens
    session = _StubSession(
        [
            AssistantMessage(
                content=[ToolUseBlock("t1", "mcp__hermes__web_search", {"query": "a"})],
                usage={
                    "input_tokens": 10,
                    "cache_read_input_tokens": 100_000,
                    "cache_creation_input_tokens": 5_000,
                    "output_tokens": 50,
                },
            ),
            UserMessage(content=[ToolResultBlock("t1", "results")]),
            AssistantMessage(
                content=[TextBlock("done")],
                usage={
                    "input_tokens": 20,
                    "cache_read_input_tokens": 140_000,
                    "cache_creation_input_tokens": 1_000,
                    "output_tokens": 30,
                },
            ),
            ResultMessage(
                result="done",
                # The SDK's turn-cumulative sum of both calls above.
                usage={
                    "input_tokens": 30,
                    "output_tokens": 80,
                    "cache_read_input_tokens": 240_000,
                    "cache_creation_input_tokens": 6_000,
                },
            ),
        ]
    )
    result, _messages = _run_turn(agent, session)

    last_call_context = 20 + 140_000 + 1_000
    turn_cumulative = 30 + 240_000 + 6_000

    assert result["last_prompt_tokens"] == last_call_context, (
        f"last_prompt_tokens={result['last_prompt_tokens']} — expected the "
        f"last call's context ({last_call_context}), not the turn sum "
        f"({turn_cumulative})"
    )
    # The compressor drives hygiene/threshold decisions — same correction.
    assert agent.context_compressor.last_prompt_tokens == last_call_context
    # Billing/session accounting stays turn-cumulative — untouched.
    assert agent.session_prompt_tokens == prompt_before + turn_cumulative
    assert result["prompt_tokens"] == turn_cumulative


def test_last_prompt_tokens_falls_back_to_cumulative_without_per_call_usage():
    """Older CLIs may not stamp per-call usage on assistant messages; the
    cumulative figure is then the only (over-)estimate available."""
    agent = _make_agent("web_search")
    session = _StubSession(
        [
            AssistantMessage(content=[TextBlock("hello")]),
            ResultMessage(
                result="hello",
                usage={"input_tokens": 12, "output_tokens": 3},
            ),
        ]
    )
    result, _messages = _run_turn(agent, session)

    assert result["last_prompt_tokens"] == 12
    assert agent.context_compressor.last_prompt_tokens == 12


# ---------------------------------------------------------------------------
# Internal iteration cap (opt-in) + graceful interrupt to end the turn
# ---------------------------------------------------------------------------


def test_max_internal_iterations_defaults_to_unlimited_when_not_configured():
    with patch(
        "hermes_cli.config.load_config_readonly",
        return_value={"claude_subscription": {"enabled": True}},
    ):
        assert claude_runtime._configured_max_internal_iterations() is None


def test_max_internal_iterations_follows_claude_subscription_config():
    with patch(
        "hermes_cli.config.load_config_readonly",
        return_value={
            "claude_subscription": {"enabled": True, "max_internal_iterations": 3}
        },
    ):
        assert claude_runtime._configured_max_internal_iterations() == 3


@pytest.mark.parametrize("raw", ["soon", 0, -1, True])
def test_max_internal_iterations_ignores_a_malformed_or_non_positive_value(raw):
    with patch(
        "hermes_cli.config.load_config_readonly",
        return_value={
            "claude_subscription": {"enabled": True, "max_internal_iterations": raw}
        },
    ):
        assert claude_runtime._configured_max_internal_iterations() is None


def test_max_internal_iterations_falls_back_to_unlimited_when_config_load_fails():
    with patch(
        "hermes_cli.config.load_config_readonly", side_effect=RuntimeError("boom")
    ):
        assert claude_runtime._configured_max_internal_iterations() is None


def test_each_assistant_message_fires_the_step_callback_once():
    agent = _make_agent("web_search")
    steps = []
    agent.step_callback = lambda iteration, prev_tools: steps.append(
        (iteration, prev_tools)
    )
    projector = ClaudeEventProjector(agent)

    projector(
        AssistantMessage(
            content=[ToolUseBlock("t1", "mcp__hermes__web_search", {"query": "a"})]
        )
    )
    projector(UserMessage(content=[ToolResultBlock("t1", "results")]))
    projector(AssistantMessage(content=[TextBlock("done")]))
    projector(ResultMessage(result="done"))

    assert [step[0] for step in steps] == [1, 2]
    # The first step has no prior iteration to report.
    assert steps[0][1] == []
    # The second step reports the first iteration's tool call, with its result.
    assert steps[1][1] == [
        {"name": "web_search", "arguments": {"query": "a"}, "result": "results"}
    ]


def test_iteration_count_stays_at_zero_without_any_assistant_message():
    agent = _make_agent("web_search")
    projector = ClaudeEventProjector(agent)
    projector(ResultMessage(result="hi"))
    assert projector.iteration_count == 0
    assert projector.iteration_cap_exceeded is False


def test_uncapped_projector_never_requests_an_interrupt():
    agent = _make_agent("web_search")
    projector = ClaudeEventProjector(agent)
    calls = []
    projector.request_interrupt = lambda: calls.append(1) or True
    for _ in range(10):
        projector(AssistantMessage(content=[TextBlock("hi")]))
    assert calls == []
    assert projector.iteration_cap_exceeded is False


def test_a_capped_projector_requests_exactly_one_interrupt_once_exceeded():
    agent = _make_agent("web_search")
    projector = ClaudeEventProjector(agent, max_internal_iterations=2)
    calls = []
    projector.request_interrupt = lambda: calls.append(1) or True

    projector(AssistantMessage(content=[TextBlock("1")]))
    assert projector.iteration_cap_exceeded is False
    projector(AssistantMessage(content=[TextBlock("2")]))
    assert projector.iteration_cap_exceeded is False
    projector(AssistantMessage(content=[TextBlock("3")]))
    assert projector.iteration_cap_exceeded is True
    # A trailing AssistantMessage after the interrupt went out must not
    # re-request it.
    projector(AssistantMessage(content=[TextBlock("4")]))

    assert calls == [1]


def test_a_missing_request_interrupt_hook_does_not_crash_the_cap_check():
    agent = _make_agent("web_search")
    projector = ClaudeEventProjector(agent, max_internal_iterations=1)
    projector(AssistantMessage(content=[TextBlock("1")]))
    projector(AssistantMessage(content=[TextBlock("2")]))
    assert projector.iteration_cap_exceeded is True


def test_run_turn_requests_a_session_interrupt_once_the_cap_is_exceeded():
    agent = _make_agent("web_search")
    session = _StubSession(
        [
            AssistantMessage(content=[TextBlock("1")]),
            AssistantMessage(content=[TextBlock("2")]),
            ResultMessage(result="2"),
        ]
    )
    with patch(
        "hermes_cli.config.load_config_readonly",
        return_value={
            "claude_subscription": {"enabled": True, "max_internal_iterations": 1}
        },
    ):
        result, _messages = _run_turn(agent, session)

    # The cap check runs on run_turn's own drain thread, so it must use the
    # nowait interrupt (blocking request_interrupt would stall that thread).
    assert session.nowait_interrupt_requests == 1
    assert session.interrupt_requests == 0
    # Hitting the cap ends the turn cleanly (a graceful Hermes-requested
    # interrupt), not a failure — it stays completed, just annotated.
    assert result["completed"] is True
    assert result["partial"] is False
    assert result["claude_iteration_cap_exceeded"] is True
    assert "iteration cap" in result["final_response"]


def test_a_capped_turns_is_error_result_is_not_treated_as_a_failed_attempt():
    """The SDK may itself flag the cap-triggered interrupt's ResultMessage as
    an error, but that interrupt was requested by Hermes, not the CLI
    choking — it must not be handed to the fallback chain as ``failed: True``
    the way an uncapped is_error result is — and it must not be reported as
    a ``completed: False`` partial turn carrying the SDK's abort text as its
    ``error`` either. The cap exemption has to hold on BOTH arms: the
    ``completed`` expression's ``is_error`` term and
    ``_record_claude_attempt``'s ``projector.error -> turn_error`` copy.
    """
    agent = _make_agent("web_search")
    session = _StubSession(
        [
            AssistantMessage(content=[TextBlock("1")]),
            AssistantMessage(content=[TextBlock("2")]),
            ResultMessage(
                subtype="error_during_execution", is_error=True, result="2"
            ),
        ]
    )
    with patch(
        "hermes_cli.config.load_config_readonly",
        return_value={
            "claude_subscription": {"enabled": True, "max_internal_iterations": 1}
        },
    ):
        result, _messages = _run_turn(agent, session)

    assert "failed" not in result
    assert result["claude_iteration_cap_exceeded"] is True
    assert result["completed"] is True
    assert result["partial"] is False
    assert result["error"] is None
    assert "iteration cap" in result["final_response"]


def test_run_turn_never_interrupts_when_the_cap_is_not_configured():
    agent = _make_agent("web_search")
    session = _StubSession(
        [
            AssistantMessage(content=[TextBlock("1")]),
            AssistantMessage(content=[TextBlock("2")]),
            ResultMessage(result="2"),
        ]
    )
    with patch(
        "hermes_cli.config.load_config_readonly",
        return_value={"claude_subscription": {"enabled": True}},
    ):
        result, _messages = _run_turn(agent, session)

    assert session.interrupt_requests == 0
    assert result["completed"] is True
    assert "claude_iteration_cap_exceeded" not in result


def test_a_capped_turn_never_triggers_an_ack_continuation_requery():
    """`iteration_cap_exceeded` must gate the ack-continuation loop on its
    own terms (not merely via `completed`, which the cap no longer forces
    false) — a turn that trips the cap on an intermediate-ack-shaped final
    reply must not spend a second `run_turn` re-querying for more."""
    agent = _make_agent("web_search")
    agent._intent_ack_continuation = True
    agent._emit_interim_assistant_message = lambda *_a, **_kw: None
    # Two internal iterations in attempt 1 (a tool round, then an ack-shaped
    # reply) so a cap of 1 is exceeded on the second one.
    first_attempt = [
        AssistantMessage(
            content=[ToolUseBlock("t1", "mcp__hermes__web_search", {"query": "a"})]
        ),
        UserMessage(content=[ToolResultBlock("t1", "results")]),
        AssistantMessage(content=[TextBlock("I'll look into the repo now.")]),
        ResultMessage(result="I'll look into the repo now."),
    ]
    session = _MultiCallSession([first_attempt, _ACK_SCRIPT])
    with patch(
        "hermes_cli.config.load_config_readonly",
        return_value={
            "claude_subscription": {"enabled": True, "max_internal_iterations": 1}
        },
    ):
        result, _messages = _run_turn_multi(agent, session)

    # Only attempt 1 ran — iteration_cap_exceeded excluded it from the
    # ack-continuation loop directly, even though it's still `completed`.
    assert session.calls == 1
    assert result["api_calls"] == 1
    assert result["completed"] is True
    assert result["claude_iteration_cap_exceeded"] is True


# ---------------------------------------------------------------------------
# Ack-continuation: a turn that only announces intent re-queries the SAME
# SDK session, mirroring the codex loop's `codex_ack_continuations < 2`.
# ---------------------------------------------------------------------------


class _MultiCallSession:
    """Like ``_StubSession`` but scripts a *different* message list per
    successive ``run_turn`` call — ack-continuation issues more than one
    ``run_turn`` against the same session object."""

    def __init__(self, scripts, *, raises_on=None):
        self.scripts = list(scripts)
        self.raises_on = raises_on or {}
        self.calls = 0
        self.prompts = []
        self.session_ids = []
        self.closed = False

    def run_turn(
        self, prompt, *, on_message, timeout=None, stall_timeout=None, stall_exempt=None
    ):
        index = self.calls
        self.calls += 1
        self.prompts.append(prompt)
        if index in self.raises_on:
            raise self.raises_on[index]
        script = self.scripts[index] if index < len(self.scripts) else []
        for message in script:
            on_message(message)
        return len(script)

    def note_session_id(self, session_id):
        self.session_ids.append(session_id)

    def request_interrupt(self):
        return True

    def request_interrupt_nowait(self):
        return True

    def close(self):
        self.closed = True


_ACK_SCRIPT = [
    AssistantMessage(content=[TextBlock("I'll look into the repo now.")]),
    ResultMessage(result="I'll look into the repo now."),
]


def _run_turn_multi(agent, session, *, user_message="hi", original_user_message="hi", messages=None):
    messages = messages if messages is not None else []
    with (
        patch.object(claude_runtime, "claude_runtime_preflight", return_value=None),
        patch.object(claude_runtime, "_ensure_session", return_value=session),
    ):
        result = run_claude_agent_sdk_turn(
            agent,
            user_message=user_message,
            original_user_message=original_user_message,
            messages=messages,
            effective_task_id="task-1",
        )
    return result, messages


def test_an_ack_only_reply_re_queries_the_same_session():
    agent = _make_agent("web_search")
    agent._intent_ack_continuation = True
    interim = []
    agent._emit_interim_assistant_message = interim.append
    session = _MultiCallSession(
        [
            _ACK_SCRIPT,
            [
                AssistantMessage(content=[TextBlock("Done, found nothing.")]),
                ResultMessage(result="Done, found nothing."),
            ],
        ]
    )
    result, messages = _run_turn_multi(agent, session)

    assert session.calls == 2
    assert session.prompts[1] == claude_runtime._ACK_CONTINUE_TEXT
    assert result["api_calls"] == 2
    assert result["final_response"] == "Done, found nothing."
    assert interim and interim[0]["content"] == "I'll look into the repo now."
    assert any(
        m.get("role") == "user" and m.get("content") == claude_runtime._ACK_CONTINUE_TEXT
        for m in messages
    )


def test_ack_continuation_caps_at_two_re_queries():
    agent = _make_agent("web_search")
    agent._intent_ack_continuation = True
    agent._emit_interim_assistant_message = lambda *_a, **_kw: None
    session = _MultiCallSession([_ACK_SCRIPT, _ACK_SCRIPT, _ACK_SCRIPT])
    result, _messages = _run_turn_multi(agent, session)

    assert session.calls == 3
    assert result["api_calls"] == 3
    assert result["final_response"] == "I'll look into the repo now."


def test_a_non_ack_reply_never_re_queries():
    agent = _make_agent("web_search")
    agent._intent_ack_continuation = True
    session = _MultiCallSession(
        [
            [
                AssistantMessage(content=[TextBlock("The answer is 42.")]),
                ResultMessage(result="The answer is 42."),
            ]
        ]
    )
    result, _messages = _run_turn_multi(agent, session)

    assert session.calls == 1
    assert result["api_calls"] == 1


def test_ack_continuation_is_off_by_default_for_claude_agent_sdk():
    """``intent_ack_continuation_mode``'s ``"auto"`` fallback is codex-only;
    the SDK path needs the same explicit opt-in any other non-codex api_mode
    does."""
    agent = _make_agent("web_search")
    session = _MultiCallSession([_ACK_SCRIPT])
    result, _messages = _run_turn_multi(agent, session)

    assert session.calls == 1
    assert result["api_calls"] == 1


def test_the_config_kill_switch_disables_continuation_even_when_opted_in():
    agent = _make_agent("web_search")
    agent._intent_ack_continuation = True
    session = _MultiCallSession([_ACK_SCRIPT])
    with patch(
        "hermes_cli.config.load_config_readonly",
        return_value={"claude_subscription": {"ack_continuation": False}},
    ):
        result, _messages = _run_turn_multi(agent, session)

    assert session.calls == 1
    assert result["api_calls"] == 1


def test_an_interrupted_attempt_never_re_queries():
    agent = _make_agent("web_search")
    agent._intent_ack_continuation = True
    agent._interrupt_requested = True
    session = _MultiCallSession([_ACK_SCRIPT])
    result, _messages = _run_turn_multi(agent, session)

    assert session.calls == 1
    assert result["interrupted"] is True


def test_an_sdk_reported_error_never_re_queries():
    agent = _make_agent("web_search")
    agent._intent_ack_continuation = True
    session = _MultiCallSession(
        [
            [
                AssistantMessage(content=[TextBlock("I'll look into the repo now.")]),
                ResultMessage(
                    result="I'll look into the repo now.",
                    is_error=True,
                    errors=["boom"],
                ),
            ]
        ]
    )
    result, _messages = _run_turn_multi(agent, session)

    assert session.calls == 1
    assert result["completed"] is False


def test_multimodal_user_input_never_re_queries():
    agent = _make_agent("web_search")
    agent._intent_ack_continuation = True
    session = _MultiCallSession([_ACK_SCRIPT])
    result, _messages = _run_turn_multi(
        agent,
        session,
        user_message=[{"type": "text", "text": "hi"}],
        original_user_message=[{"type": "text", "text": "hi"}],
    )

    assert session.calls == 1
    assert result["api_calls"] == 1


def test_a_continuation_failure_falls_back_to_the_prior_attempts_result():
    agent = _make_agent("web_search")
    agent._intent_ack_continuation = True
    agent._claude_session = session = _MultiCallSession(
        [_ACK_SCRIPT], raises_on={1: RuntimeError("boom")}
    )
    result, messages = _run_turn_multi(agent, session)

    assert session.calls == 2
    assert session.closed is True
    assert getattr(agent, "_claude_session", None) is None
    assert result["completed"] is True
    assert result["api_calls"] == 1
    assert result["final_response"] == "I'll look into the repo now."
    assert not any(
        m.get("role") == "user" and m.get("content") == claude_runtime._ACK_CONTINUE_TEXT
        for m in messages
    )


def test_a_continuation_is_error_result_falls_back_to_the_prior_attempts_result():
    """An is_error ResultMessage on the ack-continuation pass (e.g. the
    session limit landing between attempt 1 and the re-query) is the same
    situation as the continuation raising: the continuation is optional, so
    attempt 1's completed result is reported, the dangling continuation
    prompt is dropped, the limit text never lands in the transcript, and the
    wedged session is retired so the NEXT turn's first attempt respawns —
    and, if the limit persists, hands off to the fallback chain."""
    agent = _make_agent("web_search")
    agent._intent_ack_continuation = True
    agent._emit_interim_assistant_message = lambda *_a, **_kw: None
    limit_text = "You've hit your session limit · resets 6pm (Asia/Seoul)"
    agent._claude_session = session = _MultiCallSession(
        [
            _ACK_SCRIPT,
            [ResultMessage(subtype="success", is_error=True, result=limit_text)],
        ]
    )
    result, messages = _run_turn_multi(agent, session)

    assert session.calls == 2
    assert session.closed is True
    assert getattr(agent, "_claude_session", None) is None
    assert "failed" not in result
    assert result["completed"] is True
    assert result["api_calls"] == 1
    assert result["final_response"] == "I'll look into the repo now."
    assert result["error"] is None
    assert not any(limit_text in str(m.get("content", "")) for m in messages)
    assert not any(
        m.get("role") == "user" and m.get("content") == claude_runtime._ACK_CONTINUE_TEXT
        for m in messages
    )


def test_require_workspace_is_false_when_opted_in_for_all_api_modes():
    agent = _make_agent("web_search")
    agent._intent_ack_continuation = True
    captured = {}
    original = agent._looks_like_codex_intermediate_ack

    def _spy(**kwargs):
        captured.update(kwargs)
        return original(**kwargs)

    agent._looks_like_codex_intermediate_ack = _spy
    session = _MultiCallSession(
        [
            _ACK_SCRIPT,
            [AssistantMessage(content=[TextBlock("done")]), ResultMessage(result="done")],
        ]
    )
    _run_turn_multi(agent, session)

    assert captured["require_workspace"] is False
