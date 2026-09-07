"""``ClaudeSubClient`` — the one-shot OpenAI-client-shaped facade (v0.1-A).

Each ``create()`` call spins up a session, runs one turn, and tears it back
down (see the plugin README). ``SdkSession`` itself is covered by
``test_session.py``; here it is replaced with a scripted fake so these tests
pin ``client.py``'s own contract: projecting SDK messages into an
OpenAI-shaped completion, mapping ``ResultMessage`` errors through
``errors.py``, and always closing the session.

Dispatch inside the client is by SDK message class NAME (not isinstance —
see ``agent/claude_runtime.py`` for why), so the plain dataclasses from
``conftest.py`` stand in for the real SDK classes without needing
``fake_sdk``.
"""

from __future__ import annotations

from dataclasses import dataclass

import openai
import pytest


# Local stand-ins for the SDK message/block shapes — dispatch inside the
# client is by class NAME (not isinstance, see agent/claude_runtime.py), so
# these don't need to be the real SDK classes. Duplicated here rather than
# imported from conftest.py: this test directory has no __init__.py
# (AGENTS.md D13), so a relative import isn't available.
@dataclass
class TextBlock:
    text: str


@dataclass
class ThinkingBlock:
    thinking: str


@dataclass
class AssistantMessage:
    content: list
    usage: dict | None = None


@dataclass
class ResultMessage:
    is_error: bool = False
    result: str | None = None
    errors: list | None = None


def _fake_session_factory(*, messages=None, raise_exc=None, closed_flag):
    class _FakeSession:
        def __init__(self, **_kwargs):
            pass

        def run_turn(self, prompt, *, on_message, timeout=None, stall_timeout=None, stall_exempt=None):
            if raise_exc is not None:
                raise raise_exc
            for message in messages or []:
                on_message(message)
            return len(messages or [])

        def request_interrupt(self):
            return True

        def close(self):
            closed_flag["closed"] = True

    return _FakeSession


@pytest.fixture
def client_module(load_plugin_module):
    return load_plugin_module("client")


@pytest.fixture
def closed_flag():
    return {"closed": False}


def _install_fake_session(monkeypatch, client_module, closed_flag, *, messages=None, raise_exc=None):
    monkeypatch.setattr(
        client_module,
        "SdkSession",
        _fake_session_factory(messages=messages, raise_exc=raise_exc, closed_flag=closed_flag),
    )


def _make_client(client_module):
    return client_module.ClaudeSubClient(api_key="claude-sub", base_url="claude-sub://sdk")


class TestOneShotSuccess:
    def test_text_response_usage_and_finish_reason(self, monkeypatch, client_module, closed_flag):
        _install_fake_session(
            monkeypatch,
            client_module,
            closed_flag,
            messages=[
                AssistantMessage(
                    content=[TextBlock(text="Hello!")],
                    usage={
                        "input_tokens": 100,
                        "output_tokens": 20,
                        "cache_read_input_tokens": 10,
                        "cache_creation_input_tokens": 5,
                    },
                ),
                ResultMessage(is_error=False, result="Hello!"),
            ],
        )
        client = _make_client(client_module)
        completion = client.chat.completions.create(
            model="claude-sonnet-5",
            messages=[{"role": "user", "content": "hi"}],
        )
        choice = completion.choices[0]
        assert choice.message.content == "Hello!"
        assert choice.finish_reason == "stop"
        assert choice.message.tool_calls is None
        assert completion.usage.prompt_tokens == 115
        assert completion.usage.completion_tokens == 20
        assert completion.usage.total_tokens == 135
        assert completion.usage.prompt_tokens_details.cached_tokens == 10
        assert completion.model == "claude-sonnet-5"
        assert closed_flag["closed"] is True

    def test_thinking_block_becomes_reasoning(self, monkeypatch, client_module, closed_flag):
        _install_fake_session(
            monkeypatch,
            client_module,
            closed_flag,
            messages=[
                AssistantMessage(content=[ThinkingBlock(thinking="let me think"), TextBlock(text="answer")]),
                ResultMessage(is_error=False, result="answer"),
            ],
        )
        client = _make_client(client_module)
        completion = client.chat.completions.create(
            model="claude-sonnet-5", messages=[{"role": "user", "content": "hi"}]
        )
        message = completion.choices[0].message
        assert message.reasoning == "let me think"
        assert message.reasoning_content == "let me think"
        assert message.content == "answer"

    def test_no_result_message_still_returns_partial_text(self, monkeypatch, client_module, closed_flag):
        """An interrupted/dropped turn with no ResultMessage must not raise."""
        _install_fake_session(
            monkeypatch,
            client_module,
            closed_flag,
            messages=[AssistantMessage(content=[TextBlock(text="partial")])],
        )
        client = _make_client(client_module)
        completion = client.chat.completions.create(
            model="claude-sonnet-5", messages=[{"role": "user", "content": "hi"}]
        )
        assert completion.choices[0].message.content == "partial"

    def test_stream_true_yields_content_stop_and_usage_chunks(
        self, monkeypatch, client_module, closed_flag
    ):
        """v0.1-B streams real incremental chunks (see test_stream.py for the
        StreamEvent-delta path); this fake has no StreamEvent messages, so the
        content arrives as a single AssistantMessage-derived chunk, followed by
        the separate stop and usage chunks D17 requires (no longer the v0.1-A
        2-chunk fold — that collapsed shape doesn't survive real streaming)."""
        _install_fake_session(
            monkeypatch,
            client_module,
            closed_flag,
            messages=[
                AssistantMessage(content=[TextBlock(text="Hello!")]),
                ResultMessage(is_error=False, result="Hello!"),
            ],
        )
        client = _make_client(client_module)
        chunks = list(
            client.chat.completions.create(
                model="claude-sonnet-5",
                messages=[{"role": "user", "content": "hi"}],
                stream=True,
            )
        )
        assert len(chunks) == 3
        assert chunks[0].choices[0].delta.content == "Hello!"
        assert chunks[1].choices[0].finish_reason == "stop"
        assert chunks[2].usage is not None
        assert closed_flag["closed"] is True


class TestOneShotErrors:
    def test_rate_limit_result_raises_rate_limit_error(self, monkeypatch, client_module, closed_flag):
        _install_fake_session(
            monkeypatch,
            client_module,
            closed_flag,
            messages=[ResultMessage(is_error=True, result="You have hit your rate limit")],
        )
        client = _make_client(client_module)
        with pytest.raises(openai.RateLimitError):
            client.chat.completions.create(model="claude-sonnet-5", messages=[{"role": "user", "content": "hi"}])
        assert closed_flag["closed"] is True

    def test_generic_error_result_raises_503(self, monkeypatch, client_module, closed_flag):
        _install_fake_session(
            monkeypatch,
            client_module,
            closed_flag,
            messages=[ResultMessage(is_error=True, result="the CLI crashed")],
        )
        client = _make_client(client_module)
        with pytest.raises(openai.APIStatusError) as excinfo:
            client.chat.completions.create(model="claude-sonnet-5", messages=[{"role": "user", "content": "hi"}])
        assert excinfo.value.status_code == 503
        assert closed_flag["closed"] is True

    def test_stall_timeout_raises_504(self, monkeypatch, client_module, closed_flag):
        _install_fake_session(
            monkeypatch,
            client_module,
            closed_flag,
            raise_exc=TimeoutError("claude-sub turn received no SDK messages for 300s (stalled)."),
        )
        client = _make_client(client_module)
        with pytest.raises(openai.APIStatusError) as excinfo:
            client.chat.completions.create(model="claude-sonnet-5", messages=[{"role": "user", "content": "hi"}])
        assert excinfo.value.status_code == 504
        assert closed_flag["closed"] is True

    def test_sdk_exception_raises_503(self, monkeypatch, client_module, closed_flag):
        _install_fake_session(
            monkeypatch,
            client_module,
            closed_flag,
            raise_exc=RuntimeError("CLIConnectionError: could not start CLI"),
        )
        client = _make_client(client_module)
        with pytest.raises(openai.APIStatusError) as excinfo:
            client.chat.completions.create(model="claude-sonnet-5", messages=[{"role": "user", "content": "hi"}])
        assert excinfo.value.status_code == 503
        assert closed_flag["closed"] is True


class TestSessionKeyDerivation:
    def test_extra_body_session_id_is_used_verbatim(self, client_module):
        # Deriving from extra_body is a simple dict read in client.py; pin
        # the fallback hash function directly here.
        assert client_module._derive_session_key("sys", "hi") == client_module._derive_session_key(
            "sys", "hi"
        )

    def test_different_inputs_hash_differently(self, client_module):
        a = client_module._derive_session_key("sys", "hi")
        b = client_module._derive_session_key("sys", "bye")
        assert a != b
        assert len(a) == 16


class TestClientDefaults:
    def test_default_api_key_and_base_url(self, client_module):
        client = client_module.ClaudeSubClient()
        assert client.api_key == "claude-sub"
        assert client.base_url == "claude-sub://sdk"
        assert client.is_closed is False

    def test_close_marks_closed(self, client_module):
        client = client_module.ClaudeSubClient()
        client.close()
        assert client.is_closed is True
