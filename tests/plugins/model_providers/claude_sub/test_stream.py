"""``client.py`` streaming (D17): real incremental chunks, not the v0.1-A fold.

Uses the same scripted-fake-session approach as test_client_inversion.py.
``run_turn``/``continue_turn`` execute on a background thread in the real
client (``_create_stream``); the fake session here runs them synchronously
when invoked, which is fine — the thread boundary is exercised for real
(``threading.Thread`` actually starts), only the SDK's own async messaging is
faked.
"""

from __future__ import annotations

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


@dataclass
class StreamEvent:
    event: dict


def _fake_session_factory(*, messages, calls_log, closed_flag, session_module):
    class _FakeSession:
        def __init__(self, **_kwargs):
            self.closed = False
            self.interrupted = False

        def run_turn(self, prompt, *, on_message, timeout=None, stall_timeout=None, stall_exempt=None):
            calls_log.append("run_turn")
            for message in messages:
                try:
                    on_message(message)
                except session_module.PauseTurn:
                    return len(messages)
            return len(messages)

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
    """See test_client_inversion.py's fixture of the same name for why."""
    import concurrent.futures as cf

    def _fake_wait_for_pending(turn, ids):
        with turn.lock:
            matched = [entry for entry in turn.expected_ids if entry[0] in ids]
            for call_id, _name in matched:
                turn.pending.setdefault(call_id, cf.Future())
            turn.expected_ids[:] = [entry for entry in turn.expected_ids if entry[0] not in ids]

    monkeypatch.setattr(client_module, "_wait_for_pending", _fake_wait_for_pending)


def _install(monkeypatch, client_module, session_module, closed_flag, *, messages):
    calls_log: list = []
    monkeypatch.setattr(
        client_module,
        "SdkSession",
        _fake_session_factory(
            messages=messages, calls_log=calls_log, closed_flag=closed_flag, session_module=session_module
        ),
    )
    return calls_log


def _make_client(client_module):
    return client_module.ClaudeSubClient(api_key="claude-sub", base_url="claude-sub://sdk")


class TestTextOnlyStream:
    def test_text_deltas_then_result_yields_content_stop_usage_no_duplicate_text(
        self, monkeypatch, client_module, session_module, closed_flag
    ):
        messages = [
            StreamEvent(event={"type": "content_block_delta", "delta": {"type": "text_delta", "text": "Hel"}}),
            StreamEvent(event={"type": "content_block_delta", "delta": {"type": "text_delta", "text": "lo!"}}),
            AssistantMessage(content=[TextBlock(text="Hello!")], usage={"input_tokens": 10, "output_tokens": 2}),
            ResultMessage(is_error=False, result="Hello!"),
        ]
        _install(monkeypatch, client_module, session_module, closed_flag, messages=messages)
        client = _make_client(client_module)

        chunks = list(
            client.chat.completions.create(
                model="claude-sonnet-5",
                messages=[{"role": "user", "content": "hi"}],
                stream=True,
            )
        )

        # Two text deltas, no duplicate from the terminal AssistantMessage's
        # TextBlock, then a stop chunk, then a usage chunk.
        assert len(chunks) == 4
        assert chunks[0].choices[0].delta.content == "Hel"
        assert chunks[1].choices[0].delta.content == "lo!"
        assert chunks[2].choices[0].finish_reason == "stop"
        assert chunks[2].choices[0].delta.content is None
        assert chunks[3].usage.prompt_tokens == 10
        assert chunks[3].usage.completion_tokens == 2
        assert closed_flag["closed"] is True

    def test_thinking_delta_is_streamed_as_reasoning_content(
        self, monkeypatch, client_module, session_module, closed_flag
    ):
        messages = [
            StreamEvent(
                event={"type": "content_block_delta", "delta": {"type": "thinking_delta", "thinking": "hmm"}}
            ),
            AssistantMessage(content=[TextBlock(text="ok")]),
            ResultMessage(is_error=False, result="ok"),
        ]
        _install(monkeypatch, client_module, session_module, closed_flag, messages=messages)
        client = _make_client(client_module)
        chunks = list(
            client.chat.completions.create(
                model="claude-sonnet-5", messages=[{"role": "user", "content": "hi"}], stream=True
            )
        )
        assert chunks[0].choices[0].delta.reasoning_content == "hmm"
        # ok wasn't streamed as text_delta, so the AssistantMessage's TextBlock
        # is still emitted.
        assert chunks[1].choices[0].delta.content == "ok"


class TestToolCallStream:
    def test_tool_use_block_yields_tool_calls_delta_then_usage_and_pauses(
        self, monkeypatch, client_module, session_module, closed_flag, fake_sdk
    ):
        messages = [
            StreamEvent(event={"type": "content_block_delta", "delta": {"type": "text_delta", "text": "checking"}}),
            AssistantMessage(
                content=[
                    ToolUseBlock(id="toolu_1", name="mcp__hermes__read_file", input={"path": "/x"}),
                ],
                usage={"input_tokens": 5, "output_tokens": 1},
            ),
        ]
        _install(monkeypatch, client_module, session_module, closed_flag, messages=messages)
        client = _make_client(client_module)
        tools = [
            {"type": "function", "function": {"name": "read_file", "description": "", "parameters": {}}}
        ]
        chunks = list(
            client.chat.completions.create(
                model="claude-sonnet-5",
                messages=[{"role": "user", "content": "hi"}],
                tools=tools,
                stream=True,
                extra_body={"hermes_session_id": "sess-stream-tools"},
            )
        )
        assert len(chunks) == 3
        assert chunks[0].choices[0].delta.content == "checking"
        tool_delta_chunk = chunks[1]
        assert tool_delta_chunk.choices[0].finish_reason == "tool_calls"
        call = tool_delta_chunk.choices[0].delta.tool_calls[0]
        assert call.function.name == "read_file"
        assert chunks[2].usage.prompt_tokens == 5
        # The turn stays open (paused) — not closed.
        assert closed_flag["closed"] is False
