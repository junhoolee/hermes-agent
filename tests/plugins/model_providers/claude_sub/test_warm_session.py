"""Warm SDK session retention across turns (v0.1-E).

Before this card, ``client.py`` closed the underlying ``SdkSession`` (and its
CLI subprocess) the instant a turn reached ``finish_reason="stop"`` — every
follow-up user message in the same Hermes conversation paid a fresh CLI cold
start (~7-12s) plus a full bootstrapped-history reprompt (see
``test_client_oneshot.py``/``test_client_inversion.py`` for that v0.1-D
behavior, still exercised there for callers with no ``hermes_session_id``).

This file pins the v0.1-E warm-retention contract instead: a turn opened
with an explicit ``extra_body["hermes_session_id"]`` is kept ``idle`` (not
closed) after a clean stop, and a follow-up ``create()`` whose history is
*exactly* that turn's history plus new user text reuses the same
``SdkSession`` object with only the new text as the prompt — no
``<operating_instructions>``/``<prior_conversation>`` wrapping (the CLI
already holds that context). Any mismatch (system/model/reasoning/tools
changed, or the assistant history doesn't match what this session actually
said) falls back to the safe v0.1-D cold path: discard and reopen.

Uses the same scripted-fake-``SdkSession`` approach as
``test_client_inversion.py`` — see that file's factory docstring for the
"session_scripts holds one list of turns per SdkSession instantiation"
convention duplicated here (this test directory has no ``__init__.py``, so a
cross-file import isn't available — see AGENTS.md D13).
"""

from __future__ import annotations

import dataclasses
import threading
import time
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


def _fake_session_factory(*, session_scripts, calls_log, instantiations, session_module):
    """Each new ``SdkSession()`` call pops the next session script.

    A "session script" is a list of "turns" (each turn a list of SDK
    messages). Every ``run_turn``/``continue_turn`` call on that instance
    pops and drains the next turn — this is what lets a single fake session
    stand in for a real ``SdkSession`` receiving *two* separate ``query()``
    calls (a cold open, then a later warm follow-up), which is exactly the
    v0.1-E contract under test.
    """
    remaining_sessions = list(session_scripts)

    class _FakeSession:
        def __init__(self, **_kwargs):
            self._remaining = list(remaining_sessions.pop(0)) if remaining_sessions else []
            self.interrupted = False
            self.closed = False
            instantiations.append(self)

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

    return _FakeSession


@pytest.fixture
def session_module(load_plugin_module):
    return load_plugin_module("session")


@pytest.fixture
def client_module(load_plugin_module):
    return load_plugin_module("client")


@pytest.fixture
def instantiations():
    return []


@pytest.fixture(autouse=True)
def _simulate_bridge_registration(monkeypatch, client_module):
    """Bind a bridge Future directly for each block, like test_client_inversion.py.

    These tests replay a scripted, synchronous message sequence — there is no
    real async SDK MCP dispatch running concurrently, so the handler-arrival
    race that ``on_call``/hooks reconcile (test_bridge_binding.py) doesn't
    apply here; this stands in for it so the tool_calls -> continuation ->
    stop path (case (h)) still exercises client.py's real projector/turn
    bookkeeping.
    """
    import concurrent.futures as cf

    original = client_module._note_tool_blocks

    def _wrapped(turn, tool_blocks):
        sendable = original(turn, tool_blocks)
        with turn.lock:
            for block in sendable:
                call_id = getattr(block, "id", None)
                if call_id not in turn.pending:
                    turn.pending[call_id] = cf.Future()
                    turn.handled_ids.add(call_id)
        return sendable

    monkeypatch.setattr(client_module, "_note_tool_blocks", _wrapped)


def _install_fake_session(monkeypatch, client_module, session_module, instantiations, *, session_scripts):
    calls_log: list = []
    monkeypatch.setattr(
        client_module,
        "SdkSession",
        _fake_session_factory(
            session_scripts=session_scripts,
            calls_log=calls_log,
            instantiations=instantiations,
            session_module=session_module,
        ),
    )
    return calls_log


def _make_client(client_module, **settings_overrides):
    client = client_module.ClaudeSubClient(api_key="claude-sub", base_url="claude-sub://sdk")
    client._settings = dataclasses.replace(client._settings, start_timeout=0.05, **settings_overrides)
    return client


SYSTEM = {"role": "system", "content": "You are Hermes."}
FIRST_USER = {"role": "user", "content": "Remember the code word OSPREY-88."}
FIRST_REPLY_TEXT = "OK, I'll remember OSPREY-88."


def _first_call(client, *, session_id="warm-1", tools=None, model="claude-sonnet-5", reasoning_effort=None):
    return client.chat.completions.create(
        model=model,
        messages=[SYSTEM, FIRST_USER],
        tools=tools,
        reasoning_effort=reasoning_effort,
        extra_body={"hermes_session_id": session_id} if session_id else None,
    )


class TestWarmRetentionAfterStop:
    def test_stop_with_session_id_keeps_session_idle_not_closed(
        self, monkeypatch, client_module, session_module, instantiations
    ):
        script = [
            [
                [
                    AssistantMessage(content=[TextBlock(text=FIRST_REPLY_TEXT)]),
                    ResultMessage(is_error=False, result=FIRST_REPLY_TEXT),
                ]
            ]
        ]
        _install_fake_session(monkeypatch, client_module, session_module, instantiations, session_scripts=script)
        client = _make_client(client_module)

        completion = _first_call(client)
        assert completion.choices[0].finish_reason == "stop"
        assert instantiations[0].closed is False

        turn = client._turns["warm-1"]
        assert turn.state == "idle"
        assert turn.idle_timer is not None
        assert turn.seen_count == 3  # len([system, user]) + 1 assistant reply
        assert turn.last_reply_text == FIRST_REPLY_TEXT

    def test_second_create_with_matching_history_queries_only_new_text_on_same_session(
        self, monkeypatch, client_module, session_module, instantiations
    ):
        second_reply = "2+2 is 4."
        script = [
            [
                [
                    AssistantMessage(content=[TextBlock(text=FIRST_REPLY_TEXT)]),
                    ResultMessage(is_error=False, result=FIRST_REPLY_TEXT),
                ],
                [
                    AssistantMessage(content=[TextBlock(text=second_reply)]),
                    ResultMessage(is_error=False, result=second_reply),
                ],
            ]
        ]
        calls_log = _install_fake_session(
            monkeypatch, client_module, session_module, instantiations, session_scripts=script
        )
        client = _make_client(client_module)

        _first_call(client)
        assert len(instantiations) == 1
        first_prompt = calls_log[0][1]
        assert "<operating_instructions>" in first_prompt  # cold turn bootstraps full context

        second = client.chat.completions.create(
            model="claude-sonnet-5",
            messages=[
                SYSTEM,
                FIRST_USER,
                {"role": "assistant", "content": FIRST_REPLY_TEXT},
                {"role": "user", "content": "What is 2+2?"},
            ],
            extra_body={"hermes_session_id": "warm-1"},
        )
        assert second.choices[0].finish_reason == "stop"
        assert second.choices[0].message.content == second_reply

        # Same underlying SdkSession — no second SdkSession() instantiation.
        assert len(instantiations) == 1
        assert calls_log[-1][0] == "run_turn"
        second_prompt = calls_log[-1][1]
        assert second_prompt == "What is 2+2?"
        assert "<operating_instructions>" not in second_prompt
        assert "<prior_conversation>" not in second_prompt
        assert FIRST_REPLY_TEXT not in second_prompt


class TestColdFallbackOnMismatch:
    def _two_session_script(self, second_reply="fresh answer"):
        return [
            [
                [
                    AssistantMessage(content=[TextBlock(text=FIRST_REPLY_TEXT)]),
                    ResultMessage(is_error=False, result=FIRST_REPLY_TEXT),
                ]
            ],
            [
                [
                    AssistantMessage(content=[TextBlock(text=second_reply)]),
                    ResultMessage(is_error=False, result=second_reply),
                ]
            ],
        ]

    def test_system_text_change_forces_cold_new_session(
        self, monkeypatch, client_module, session_module, instantiations
    ):
        calls_log = _install_fake_session(
            monkeypatch,
            client_module,
            session_module,
            instantiations,
            session_scripts=self._two_session_script(),
        )
        client = _make_client(client_module)
        _first_call(client)

        second = client.chat.completions.create(
            model="claude-sonnet-5",
            messages=[
                {"role": "system", "content": "You are a DIFFERENT assistant now."},
                FIRST_USER,
                {"role": "assistant", "content": FIRST_REPLY_TEXT},
                {"role": "user", "content": "anything"},
            ],
            extra_body={"hermes_session_id": "warm-1"},
        )
        assert second.choices[0].finish_reason == "stop"
        assert len(instantiations) == 2
        assert instantiations[0].closed is True  # old session torn down
        assert calls_log[-1][0] == "run_turn"
        assert "<operating_instructions>" in calls_log[-1][1]  # full cold rebuild

    def test_model_change_forces_cold_new_session(
        self, monkeypatch, client_module, session_module, instantiations
    ):
        _install_fake_session(
            monkeypatch,
            client_module,
            session_module,
            instantiations,
            session_scripts=self._two_session_script(),
        )
        client = _make_client(client_module)
        _first_call(client, model="claude-sonnet-5")

        client.chat.completions.create(
            model="claude-opus-5",
            messages=[
                SYSTEM,
                FIRST_USER,
                {"role": "assistant", "content": FIRST_REPLY_TEXT},
                {"role": "user", "content": "anything"},
            ],
            extra_body={"hermes_session_id": "warm-1"},
        )
        assert len(instantiations) == 2
        assert instantiations[0].closed is True

    def test_tools_change_forces_cold_new_session(
        self, monkeypatch, client_module, session_module, instantiations
    ):
        tools_a = [{"type": "function", "function": {"name": "read_file", "parameters": {}}}]
        tools_b = [{"type": "function", "function": {"name": "terminal", "parameters": {}}}]
        _install_fake_session(
            monkeypatch,
            client_module,
            session_module,
            instantiations,
            session_scripts=self._two_session_script(),
        )
        client = _make_client(client_module)
        _first_call(client, tools=tools_a)

        client.chat.completions.create(
            model="claude-sonnet-5",
            messages=[
                SYSTEM,
                FIRST_USER,
                {"role": "assistant", "content": FIRST_REPLY_TEXT},
                {"role": "user", "content": "anything"},
            ],
            tools=tools_b,
            extra_body={"hermes_session_id": "warm-1"},
        )
        assert len(instantiations) == 2
        assert instantiations[0].closed is True

    def test_assistant_text_mismatch_forces_cold(
        self, monkeypatch, client_module, session_module, instantiations
    ):
        """E.g. compaction rewrote history — the reply on record no longer matches."""
        _install_fake_session(
            monkeypatch,
            client_module,
            session_module,
            instantiations,
            session_scripts=self._two_session_script(),
        )
        client = _make_client(client_module)
        _first_call(client)

        client.chat.completions.create(
            model="claude-sonnet-5",
            messages=[
                SYSTEM,
                FIRST_USER,
                {"role": "assistant", "content": "a completely different, compacted reply"},
                {"role": "user", "content": "anything"},
            ],
            extra_body={"hermes_session_id": "warm-1"},
        )
        assert len(instantiations) == 2
        assert instantiations[0].closed is True


class TestNoWarmWithoutSessionId:
    def test_no_session_id_closes_immediately_on_stop(
        self, monkeypatch, client_module, session_module, instantiations
    ):
        script = [
            [
                [
                    AssistantMessage(content=[TextBlock(text=FIRST_REPLY_TEXT)]),
                    ResultMessage(is_error=False, result=FIRST_REPLY_TEXT),
                ]
            ]
        ]
        _install_fake_session(monkeypatch, client_module, session_module, instantiations, session_scripts=script)
        client = _make_client(client_module)

        client.chat.completions.create(model="claude-sonnet-5", messages=[SYSTEM, FIRST_USER])
        assert instantiations[0].closed is True
        assert client._turns == {}


class TestIdleTtlDisabled:
    def test_idle_ttl_zero_closes_immediately(
        self, monkeypatch, client_module, session_module, instantiations
    ):
        script = [
            [
                [
                    AssistantMessage(content=[TextBlock(text=FIRST_REPLY_TEXT)]),
                    ResultMessage(is_error=False, result=FIRST_REPLY_TEXT),
                ]
            ]
        ]
        _install_fake_session(monkeypatch, client_module, session_module, instantiations, session_scripts=script)
        client = _make_client(client_module, idle_session_ttl=0.0)

        _first_call(client)
        assert instantiations[0].closed is True
        assert client._turns == {}


class TestIdleTimerExpiry:
    def test_idle_timer_expiry_closes_and_removes_turn(
        self, monkeypatch, client_module, session_module, instantiations
    ):
        script = [
            [
                [
                    AssistantMessage(content=[TextBlock(text=FIRST_REPLY_TEXT)]),
                    ResultMessage(is_error=False, result=FIRST_REPLY_TEXT),
                ]
            ]
        ]
        _install_fake_session(monkeypatch, client_module, session_module, instantiations, session_scripts=script)
        client = _make_client(client_module, idle_session_ttl=0.05)

        _first_call(client)
        assert instantiations[0].closed is False
        assert "warm-1" in client._turns

        time.sleep(0.3)
        assert instantiations[0].closed is True
        assert client._turns == {}

    def test_followup_before_expiry_cancels_timer_and_goes_warm(
        self, monkeypatch, client_module, session_module, instantiations
    ):
        second_reply = "still here"
        script = [
            [
                [
                    AssistantMessage(content=[TextBlock(text=FIRST_REPLY_TEXT)]),
                    ResultMessage(is_error=False, result=FIRST_REPLY_TEXT),
                ],
                [
                    AssistantMessage(content=[TextBlock(text=second_reply)]),
                    ResultMessage(is_error=False, result=second_reply),
                ],
            ]
        ]
        _install_fake_session(monkeypatch, client_module, session_module, instantiations, session_scripts=script)
        client = _make_client(client_module, idle_session_ttl=1800.0)

        _first_call(client)
        turn = client._turns["warm-1"]
        armed_timer = turn.idle_timer
        assert armed_timer is not None

        client.chat.completions.create(
            model="claude-sonnet-5",
            messages=[
                SYSTEM,
                FIRST_USER,
                {"role": "assistant", "content": FIRST_REPLY_TEXT},
                {"role": "user", "content": "still there?"},
            ],
            extra_body={"hermes_session_id": "warm-1"},
        )
        assert armed_timer.finished.is_set()  # cancel() flips this synchronously
        assert len(instantiations) == 1
        assert instantiations[0].closed is False


class TestWarmFollowupToolFlow:
    def test_tool_calls_continuation_stop_on_warm_followup(
        self, monkeypatch, client_module, session_module, instantiations
    ):
        """v0.1-D's tool_calls -> continuation -> stop dance still works once warm."""
        tools = [{"type": "function", "function": {"name": "read_file", "parameters": {}}}]
        script = [
            [
                # Turn 1: cold open, plain text stop (goes idle).
                [
                    AssistantMessage(content=[TextBlock(text=FIRST_REPLY_TEXT)]),
                    ResultMessage(is_error=False, result=FIRST_REPLY_TEXT),
                ],
                # Turn 2: warm follow-up, pauses on a tool call.
                [
                    AssistantMessage(
                        content=[
                            ToolUseBlock(id="toolu_1", name="mcp__hermes__read_file", input={"path": "/x"})
                        ]
                    ),
                ],
                # Turn 2 continuation: resolves to stop.
                [
                    AssistantMessage(content=[TextBlock(text="the file says hi")]),
                    ResultMessage(is_error=False, result="the file says hi"),
                ],
            ]
        ]
        calls_log = _install_fake_session(
            monkeypatch, client_module, session_module, instantiations, session_scripts=script
        )
        client = _make_client(client_module)

        _first_call(client, tools=tools)

        warm = client.chat.completions.create(
            model="claude-sonnet-5",
            messages=[
                SYSTEM,
                FIRST_USER,
                {"role": "assistant", "content": FIRST_REPLY_TEXT},
                {"role": "user", "content": "read /x for me"},
            ],
            tools=tools,
            extra_body={"hermes_session_id": "warm-1"},
        )
        assert warm.choices[0].finish_reason == "tool_calls"
        call = warm.choices[0].message.tool_calls[0]
        turn = client._turns["warm-1"]
        assert turn.state == "paused"
        assert len(instantiations) == 1  # still the same warm session, no new subprocess

        final = client.chat.completions.create(
            model="claude-sonnet-5",
            messages=[
                SYSTEM,
                FIRST_USER,
                {"role": "assistant", "content": FIRST_REPLY_TEXT},
                {"role": "user", "content": "read /x for me"},
                {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": call.id,
                            "type": "function",
                            "function": {"name": "read_file", "arguments": call.function.arguments},
                        }
                    ],
                },
                {"role": "tool", "tool_call_id": call.id, "content": "hi"},
            ],
            tools=tools,
            extra_body={"hermes_session_id": "warm-1"},
        )
        assert final.choices[0].finish_reason == "stop"
        assert final.choices[0].message.content == "the file says hi"
        assert calls_log[-1] == ("continue_turn", None)
        assert len(instantiations) == 1
        assert instantiations[0].closed is False
        assert client._turns["warm-1"].state == "idle"


class TestWarmFollowupWithImage:
    """v0.1-G: a warm follow-up whose new message carries an image still reuses the session."""

    def test_image_in_new_message_produces_stream_prompt_on_same_session(
        self, monkeypatch, client_module, session_module, instantiations, load_plugin_module
    ):
        convert = load_plugin_module("convert")
        second_reply = "I see it"
        script = [
            [
                [
                    AssistantMessage(content=[TextBlock(text=FIRST_REPLY_TEXT)]),
                    ResultMessage(is_error=False, result=FIRST_REPLY_TEXT),
                ],
                [
                    AssistantMessage(content=[TextBlock(text=second_reply)]),
                    ResultMessage(is_error=False, result=second_reply),
                ],
            ]
        ]
        calls_log = _install_fake_session(
            monkeypatch, client_module, session_module, instantiations, session_scripts=script
        )
        client = _make_client(client_module)
        _first_call(client)

        second = client.chat.completions.create(
            model="claude-sonnet-5",
            messages=[
                SYSTEM,
                FIRST_USER,
                {"role": "assistant", "content": FIRST_REPLY_TEXT},
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "what's this?"},
                        {"type": "image_url", "image_url": {"url": "data:image/png;base64,AAAA"}},
                    ],
                },
            ],
            extra_body={"hermes_session_id": "warm-1"},
        )
        assert second.choices[0].finish_reason == "stop"
        assert second.choices[0].message.content == second_reply

        # Same underlying SdkSession — no second SdkSession() instantiation
        # (warm detection still works via the assistant-text comparison,
        # which is unaffected by the new message's content type).
        assert len(instantiations) == 1
        assert calls_log[-1][0] == "run_turn"
        prompt = calls_log[-1][1]
        assert isinstance(prompt, convert.StreamPrompt)
        assert prompt.image_count == 1


class TestClientCloseCleansIdleSessions:
    def test_close_cancels_idle_timers_and_closes_sessions(
        self, monkeypatch, client_module, session_module, instantiations
    ):
        script = [
            [
                [
                    AssistantMessage(content=[TextBlock(text=FIRST_REPLY_TEXT)]),
                    ResultMessage(is_error=False, result=FIRST_REPLY_TEXT),
                ]
            ]
        ]
        _install_fake_session(monkeypatch, client_module, session_module, instantiations, session_scripts=script)
        client = _make_client(client_module)

        _first_call(client)
        turn = client._turns["warm-1"]
        idle_timer = turn.idle_timer
        assert idle_timer is not None

        client.close()
        assert idle_timer.finished.is_set()  # cancel() flips this synchronously
        assert instantiations[0].closed is True
        assert client._turns == {}
        assert client.is_closed is True


class TestStreamingWarmSession:
    def test_streaming_stop_keeps_idle_then_second_stream_is_warm(
        self, monkeypatch, client_module, session_module, instantiations
    ):
        second_reply = "still streaming warm"
        script = [
            [
                [
                    AssistantMessage(content=[TextBlock(text=FIRST_REPLY_TEXT)]),
                    ResultMessage(is_error=False, result=FIRST_REPLY_TEXT),
                ],
                [
                    AssistantMessage(content=[TextBlock(text=second_reply)]),
                    ResultMessage(is_error=False, result=second_reply),
                ],
            ]
        ]
        calls_log = _install_fake_session(
            monkeypatch, client_module, session_module, instantiations, session_scripts=script
        )
        client = _make_client(client_module)

        first_chunks = list(
            client.chat.completions.create(
                model="claude-sonnet-5",
                messages=[SYSTEM, FIRST_USER],
                stream=True,
                extra_body={"hermes_session_id": "warm-1"},
            )
        )
        assert first_chunks[-2].choices[0].finish_reason == "stop"
        assert instantiations[0].closed is False
        assert client._turns["warm-1"].state == "idle"

        second_chunks = list(
            client.chat.completions.create(
                model="claude-sonnet-5",
                messages=[
                    SYSTEM,
                    FIRST_USER,
                    {"role": "assistant", "content": FIRST_REPLY_TEXT},
                    {"role": "user", "content": "keep going"},
                ],
                stream=True,
                extra_body={"hermes_session_id": "warm-1"},
            )
        )
        assert second_chunks[0].choices[0].delta.content == second_reply
        assert len(instantiations) == 1
        assert calls_log[-1] == ("run_turn", "keep going")


class TestIdleExpiryRaceAgainstWarmClaim:
    def test_idle_timer_firing_during_warm_decision_does_not_close_reused_session(
        self, monkeypatch, client_module, session_module, instantiations
    ):
        """D35 review (run 180): before this fix, ``create()`` fetched the
        turn under ``_turns_lock``, released it, then decided "warm" and set
        ``turn.state = "open"`` several lines later — all outside the lock.
        The idle-timer callback (also gated on ``state == "idle"``, but under
        the lock) could fire in that gap, pop the turn, and close its
        session before ``create()`` got to claim it, so the warm follow-up
        would go on to call ``run_turn``/``continue_turn`` on an already-closed
        session.

        This test forces exactly that interleaving: from inside
        ``_is_warm_followup`` (which the fix now calls *while holding*
        ``_turns_lock``), it spawns a thread that invokes the idle timer's
        callback directly (bypassing the real TTL wait, and bypassing
        ``Timer.cancel()``'s usual protection too — this reproduces the
        documented edge case where a timer's background thread has already
        passed its own cancellation check). That thread can only actually
        resolve its lock-guarded check once the decision block releases
        ``_turns_lock``, so this proves the fix serializes the two sides
        instead of merely hoping they don't interleave.

        The fake session's run_turn/continue_turn are pure in-memory
        replays with no real I/O, so after the lock is released the rest of
        create() (run the warm turn, stop, re-idle) can finish before the OS
        ever schedules the blocked racer thread back in — which would let
        the racer's stale fire land during a *later*, unrelated idle period
        instead of the window under test. ``_cancel_idle_timer`` is the
        first thing create() does after releasing the lock, before it ever
        calls run_turn, so pinning it as a rendezvous (wait for the racer to
        resolve before letting the call proceed) keeps the race deterministic
        without changing what's being exercised.
        """
        second_reply = "still here after the race"
        script = [
            [
                [
                    AssistantMessage(content=[TextBlock(text=FIRST_REPLY_TEXT)]),
                    ResultMessage(is_error=False, result=FIRST_REPLY_TEXT),
                ],
                [
                    AssistantMessage(content=[TextBlock(text=second_reply)]),
                    ResultMessage(is_error=False, result=second_reply),
                ],
            ]
        ]
        _install_fake_session(monkeypatch, client_module, session_module, instantiations, session_scripts=script)
        client = _make_client(client_module, idle_session_ttl=1800.0)

        _first_call(client)
        turn = client._turns["warm-1"]
        assert turn.idle_timer is not None
        stale_idle_timer = turn.idle_timer

        fired = threading.Event()
        original_is_warm_followup = client_module._is_warm_followup

        def _patched_is_warm_followup(*args, **kwargs):
            def _fire_idle_expiry() -> None:
                stale_idle_timer.function()  # simulate the real Timer firing now
                fired.set()

            racer = threading.Thread(target=_fire_idle_expiry, daemon=True)
            racer.start()
            # Give the racer a chance to reach (and block on) _turns_lock,
            # which this call is currently holding.
            time.sleep(0.05)
            return original_is_warm_followup(*args, **kwargs)

        monkeypatch.setattr(client_module, "_is_warm_followup", _patched_is_warm_followup)

        original_cancel_idle_timer = client._cancel_idle_timer

        def _patched_cancel_idle_timer(t):
            assert fired.wait(timeout=2.0)
            return original_cancel_idle_timer(t)

        monkeypatch.setattr(client, "_cancel_idle_timer", _patched_cancel_idle_timer)

        second = client.chat.completions.create(
            model="claude-sonnet-5",
            messages=[
                SYSTEM,
                FIRST_USER,
                {"role": "assistant", "content": FIRST_REPLY_TEXT},
                {"role": "user", "content": "still there?"},
            ],
            extra_body={"hermes_session_id": "warm-1"},
        )

        assert second.choices[0].finish_reason == "stop"
        assert second.choices[0].message.content == second_reply
        assert len(instantiations) == 1  # no cold-restart session was opened
        assert instantiations[0].closed is False  # the racer's close() was a no-op
