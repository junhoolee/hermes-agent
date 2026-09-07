"""``client.py`` tool-call inversion (D16) against a scripted fake SDK session.

``client_module.SdkSession`` is monkeypatched with a fake that drives a
*scripted* sequence of SDK messages through the real ``on_message`` callback
client.py installs (``_InversionProjector``), including honoring
``PauseTurn`` the same way the real ``session.py`` does — so these tests pin
client.py's own inversion contract (expected-id matching, Future
resolution, tool_calls/usage shaping, continuation vs. new-turn
classification) without needing the real ``claude_agent_sdk``.
"""

from __future__ import annotations

import dataclasses
import json
from dataclasses import dataclass

import pytest


@dataclass
class TextBlock:
    text: str


@dataclass
class ToolUseBlock:
    id: str
    name: str
    input: dict


@dataclass
class AssistantMessage:
    content: list
    usage: dict | None = None


@dataclass
class ResultMessage:
    is_error: bool = False
    result: str | None = None
    errors: list | None = None


def _fake_session_factory(*, session_scripts, calls_log, closed_flag, session_module):
    """*session_scripts* is a list of per-``SdkSession`` scripts.

    Each new ``SdkSession()`` instantiation (a fresh underlying CLI
    subprocess in reality) pops the next entry — itself a list of "turns"
    (each turn a list of SDK messages). The first turn of a session is
    delivered by ``run_turn``; each subsequent turn by the next
    ``continue_turn`` call — mirroring how the real session pauses on a
    tool-call boundary and resumes on continuation, without needing
    threads/asyncio.
    """
    remaining_sessions = list(session_scripts)

    class _FakeSession:
        def __init__(self, **_kwargs):
            self._remaining = list(remaining_sessions.pop(0)) if remaining_sessions else []
            self.interrupted = False
            self.closed = False

        def _drain_next_turn(self, on_message):
            if not self._remaining:
                return 0
            messages = self._remaining.pop(0)
            delivered = 0
            for message in messages:
                delivered += 1
                try:
                    on_message(message)
                except session_module.PauseTurn:
                    return delivered
            return delivered

        def run_turn(self, prompt, *, on_message, timeout=None, stall_timeout=None, stall_exempt=None):
            calls_log.append(("run_turn", prompt))
            return self._drain_next_turn(on_message)

        def continue_turn(self, *, on_message, timeout=None, stall_timeout=None, stall_exempt=None):
            calls_log.append(("continue_turn", None))
            return self._drain_next_turn(on_message)

        def request_interrupt(self):
            self.interrupted = True
            return True

        def close(self):
            self.closed = True
            closed_flag["closed"] = True

    return _FakeSession


@pytest.fixture
def session_module(load_plugin_module):
    return load_plugin_module("session")


@pytest.fixture
def client_module(load_plugin_module):
    return load_plugin_module("client")


@pytest.fixture
def closed_flag():
    return {"closed": False}


@pytest.fixture(autouse=True)
def _simulate_bridge_registration(monkeypatch, client_module):
    """Stand in for the real async bridge handler calling ``on_call``.

    These tests replay a scripted, synchronous message sequence straight
    through ``on_message`` — there is no real ``claude_agent_sdk`` MCP
    dispatch running concurrently to actually invoke a bridge handler (that
    path, including the real expected-id/name matching inside ``on_call``,
    is covered by test_bridge.py's handler tests). Here, once
    ``_register_expected`` has queued the expected ids for the AssistantMessage
    just seen, register a Future for each directly — exercising client.py's
    own contract (pump/classification/response shaping) without needing
    threads or the real SDK.
    """
    import concurrent.futures as cf

    def _fake_wait_for_pending(turn, ids):
        with turn.lock:
            matched = [entry for entry in turn.expected_ids if entry[0] in ids]
            for call_id, _name in matched:
                turn.pending.setdefault(call_id, cf.Future())
            turn.expected_ids[:] = [entry for entry in turn.expected_ids if entry[0] not in ids]

    monkeypatch.setattr(client_module, "_wait_for_pending", _fake_wait_for_pending)


def _install_fake_session(monkeypatch, client_module, session_module, closed_flag, *, script=None, session_scripts=None):
    """*script* (one session's turns) or *session_scripts* (multiple sessions)."""
    if session_scripts is None:
        session_scripts = [script or []]
    calls_log: list = []
    monkeypatch.setattr(
        client_module,
        "SdkSession",
        _fake_session_factory(
            session_scripts=session_scripts,
            calls_log=calls_log,
            closed_flag=closed_flag,
            session_module=session_module,
        ),
    )
    return calls_log


def _make_client(client_module):
    """A client with a short ``start_timeout``.

    These tests replay scripted SDK messages directly through
    ``on_message`` without a real bridge handler ever calling ``on_call``
    (that async-dispatch path is covered separately by test_bridge.py), so
    ``_wait_for_pending`` always exhausts its wait here — keep the bound
    short so that's milliseconds, not the real 60s default.
    """
    client = client_module.ClaudeSubClient(api_key="claude-sub", base_url="claude-sub://sdk")
    client._settings = dataclasses.replace(client._settings, start_timeout=0.05)
    return client


READ_FILE_TOOL = [
    {
        "type": "function",
        "function": {"name": "read_file", "description": "Read a file.", "parameters": {}},
    }
]

TWO_TOOLS = READ_FILE_TOOL + [
    {
        "type": "function",
        "function": {"name": "terminal", "description": "Run a command.", "parameters": {}},
    }
]


class TestSingleToolRoundTrip:
    def test_tool_call_then_continuation_resolves_to_stop(
        self, monkeypatch, client_module, session_module, closed_flag
    ):
        script = [
            [
                AssistantMessage(
                    content=[
                        TextBlock(text="Let me check."),
                        ToolUseBlock(
                            id="toolu_1", name="mcp__hermes__read_file", input={"path": "/tmp/x"}
                        ),
                    ],
                    usage={"input_tokens": 50, "output_tokens": 5},
                ),
            ],
            [
                AssistantMessage(content=[TextBlock(text="It says hello.")], usage={"input_tokens": 60, "output_tokens": 8}),
                ResultMessage(is_error=False, result="It says hello."),
            ],
        ]
        calls_log = _install_fake_session(
            monkeypatch, client_module, session_module, closed_flag, script=script
        )
        client = _make_client(client_module)

        first = client.chat.completions.create(
            model="claude-sonnet-5",
            messages=[{"role": "user", "content": "read /tmp/x"}],
            tools=READ_FILE_TOOL,
            extra_body={"hermes_session_id": "sess-1"},
        )
        choice = first.choices[0]
        assert choice.finish_reason == "tool_calls"
        assert len(choice.message.tool_calls) == 1
        call = choice.message.tool_calls[0]
        assert call.function.name == "read_file"
        assert json.loads(call.function.arguments) == {"path": "/tmp/x"}
        assert choice.message.content == "Let me check."
        assert calls_log[0][0] == "run_turn"
        assert closed_flag["closed"] is False  # turn stays open

        second = client.chat.completions.create(
            model="claude-sonnet-5",
            messages=[
                {"role": "user", "content": "read /tmp/x"},
                {
                    "role": "assistant",
                    "content": "Let me check.",
                    "tool_calls": [
                        {
                            "id": call.id,
                            "type": "function",
                            "function": {"name": "read_file", "arguments": call.function.arguments},
                        }
                    ],
                },
                {"role": "tool", "tool_call_id": call.id, "content": "file says: hello"},
            ],
            tools=READ_FILE_TOOL,
            extra_body={"hermes_session_id": "sess-1"},
        )
        choice2 = second.choices[0]
        assert choice2.finish_reason == "stop"
        assert choice2.message.content == "It says hello."
        assert choice2.message.tool_calls is None
        assert second.usage.prompt_tokens == 60
        assert second.usage.completion_tokens == 8
        assert calls_log[-1] == ("continue_turn", None)
        assert closed_flag["closed"] is True

    def test_parallel_tool_calls_both_resolved_in_one_continuation(
        self, monkeypatch, client_module, session_module, closed_flag
    ):
        script = [
            [
                AssistantMessage(
                    content=[
                        ToolUseBlock(id="a", name="mcp__hermes__read_file", input={"path": "1"}),
                        ToolUseBlock(id="b", name="mcp__hermes__terminal", input={"command": "ls"}),
                    ],
                ),
            ],
            [
                AssistantMessage(content=[TextBlock(text="done")]),
                ResultMessage(is_error=False, result="done"),
            ],
        ]
        _install_fake_session(monkeypatch, client_module, session_module, closed_flag, script=script)
        client = _make_client(client_module)

        first = client.chat.completions.create(
            model="claude-sonnet-5",
            messages=[{"role": "user", "content": "go"}],
            tools=TWO_TOOLS,
            extra_body={"hermes_session_id": "sess-parallel"},
        )
        calls = first.choices[0].message.tool_calls
        assert len(calls) == 2
        names = {c.function.name for c in calls}
        assert names == {"read_file", "terminal"}

        tool_messages = [
            {"role": "tool", "tool_call_id": c.id, "content": f"result-for-{c.function.name}"}
            for c in calls
        ]
        assistant_echo = {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {"id": c.id, "type": "function", "function": {"name": c.function.name, "arguments": c.function.arguments}}
                for c in calls
            ],
        }
        second = client.chat.completions.create(
            model="claude-sonnet-5",
            messages=[{"role": "user", "content": "go"}, assistant_echo, *tool_messages],
            tools=TWO_TOOLS,
            extra_body={"hermes_session_id": "sess-parallel"},
        )
        assert second.choices[0].finish_reason == "stop"
        assert second.choices[0].message.content == "done"


class TestNonContinuationSupersedesOpenTurn:
    def test_new_request_interrupts_and_cancels_previous_open_turn(
        self, monkeypatch, client_module, session_module, closed_flag
    ):
        # Two separate underlying SdkSessions: the first opens with a tool
        # call and is abandoned; the second (a fresh CLI subprocess, in
        # reality) answers the new user turn directly.
        session_scripts = [
            [
                [AssistantMessage(content=[ToolUseBlock(id="a", name="mcp__hermes__read_file", input={})])],
            ],
            [
                [
                    AssistantMessage(content=[TextBlock(text="fresh answer")]),
                    ResultMessage(is_error=False, result="fresh answer"),
                ],
            ],
        ]
        _install_fake_session(
            monkeypatch, client_module, session_module, closed_flag, session_scripts=session_scripts
        )
        client = _make_client(client_module)

        first = client.chat.completions.create(
            model="claude-sonnet-5",
            messages=[{"role": "user", "content": "go"}],
            tools=READ_FILE_TOOL,
            extra_body={"hermes_session_id": "sess-new"},
        )
        assert first.choices[0].finish_reason == "tool_calls"
        pending_future = list(client._turns["sess-new"].pending.values())[0]

        # A brand-new user turn (no matching tool-result tail) instead of a
        # continuation: the old open turn must be interrupted and its
        # pending Future cancelled, and a fresh turn started.
        second = client.chat.completions.create(
            model="claude-sonnet-5",
            messages=[{"role": "user", "content": "actually, something else"}],
            tools=READ_FILE_TOOL,
            extra_body={"hermes_session_id": "sess-new"},
        )
        assert pending_future.cancelled()
        assert second.choices[0].finish_reason == "stop"
        assert second.choices[0].message.content == "fresh answer"


class TestUsageAccounting:
    def test_usage_reflects_last_assistant_message(
        self, monkeypatch, client_module, session_module, closed_flag
    ):
        script = [
            [
                AssistantMessage(content=[TextBlock(text="hi")], usage={"input_tokens": 999, "output_tokens": 999}),
                ResultMessage(is_error=False, result="hi"),
            ],
        ]
        _install_fake_session(monkeypatch, client_module, session_module, closed_flag, script=script)
        client = _make_client(client_module)
        completion = client.chat.completions.create(
            model="claude-sonnet-5", messages=[{"role": "user", "content": "hi"}]
        )
        assert completion.usage.prompt_tokens == 999
        assert completion.usage.completion_tokens == 999
