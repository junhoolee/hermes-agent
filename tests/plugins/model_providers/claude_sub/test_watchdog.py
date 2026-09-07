"""D18 watchdogs: silence (stall), pending-exemption, and open-turn (orphan).

Two levels are exercised:

* ``session.py``'s own stall detection — real ``SdkSession`` driven by the
  ``_FakeAsyncClient`` pattern from test_session.py (a genuine event-loop
  thread, real timing), pinning that a stall now calls
  ``request_interrupt()`` before raising (the card A follow-up-note gap this
  card closes) and that ``stall_exempt`` genuinely keeps a turn alive.
* ``client.py``'s open-turn (orphan) watchdog — a scripted fake session (the
  test_client_inversion.py pattern), waited on via ``Timer.join()`` so the
  assertion is event-based rather than a blind sleep (AGENTS.md flake
  policy: no ``assert not _wait_until(...)``-style timing races).
"""

from __future__ import annotations

import asyncio
import threading

import pytest


# ---------------------------------------------------------------------------
# session.py-level: stall detection + interrupt + stall_exempt.
# ---------------------------------------------------------------------------


class _FakeAsyncClient:
    def __init__(self, *, options):
        self.options = options
        self.connected = False
        self.interrupted = False
        self.queries: list[str] = []
        self.messages_factory = lambda: iter(())
        self._message_delay = 0.0

    async def connect(self):
        self.connected = True

    async def disconnect(self):
        self.connected = False

    async def query(self, prompt: str):
        self.queries.append(prompt)

    def receive_messages(self):
        messages = list(self.messages_factory())
        delay = self._message_delay

        async def _gen():
            for message in messages:
                if delay:
                    await asyncio.sleep(delay)
                yield message

        return _gen()

    async def interrupt(self):
        self.interrupted = True


@pytest.fixture
def fake_client_factory():
    holder: list[_FakeAsyncClient] = []

    def factory(*, options):
        client = _FakeAsyncClient(options=options)
        holder.append(client)
        return client

    return factory, holder


def _session(load_plugin_module, factory):
    session_mod = load_plugin_module("session")
    return session_mod.SdkSession(
        options_factory=lambda: object(),
        client_factory=factory,
        transport_factory=None,
        start_timeout=5.0,
        close_timeout=5.0,
    )


class TestStallWatchdog:
    def test_stall_without_exemption_interrupts_then_raises(
        self, load_plugin_module, fake_client_factory
    ):
        factory, holder = fake_client_factory
        session = _session(load_plugin_module, factory)
        session.ensure_started()
        holder[0].messages_factory = lambda: ["m1", "m2 (never arrives in time)"]
        holder[0]._message_delay = 2.0  # far longer than stall_timeout below
        try:
            with pytest.raises(TimeoutError):
                session.run_turn(
                    "hello", on_message=lambda _m: None, timeout=10.0, stall_timeout=0.2
                )
            assert holder[0].interrupted is True
        finally:
            session.close()

    def test_stall_exempt_keeps_turn_alive_past_stall_timeout(
        self, load_plugin_module, fake_client_factory
    ):
        """A pending bridge tool call (stall_exempt() -> True) must not trip
        the silence watchdog even though the next message arrives well after
        stall_timeout — it just has to arrive before the overall timeout."""
        factory, holder = fake_client_factory
        session = _session(load_plugin_module, factory)
        session.ensure_started()
        holder[0].messages_factory = lambda: ["m1"]
        holder[0]._message_delay = 0.3
        received: list = []
        try:
            delivered = session.run_turn(
                "hello",
                on_message=received.append,
                timeout=5.0,
                stall_timeout=0.05,
                stall_exempt=lambda: True,
            )
            assert delivered == 1
            assert received == ["m1"]
            assert holder[0].interrupted is False
        finally:
            session.close()


# ---------------------------------------------------------------------------
# client.py-level: the open-turn (orphan) watchdog.
# ---------------------------------------------------------------------------


def _fake_session_factory(*, first_turn, session_module):
    class _FakeSession:
        def __init__(self, **_kwargs):
            self._delivered_first_turn = False
            self.interrupted = False
            self.closed = False

        def run_turn(self, prompt, *, on_message, timeout=None, stall_timeout=None, stall_exempt=None):
            self._delivered_first_turn = True
            for message in first_turn:
                try:
                    on_message(message)
                except session_module.PauseTurn:
                    return len(first_turn)
            return len(first_turn)

        def continue_turn(self, *, on_message, timeout=None, stall_timeout=None, stall_exempt=None):
            raise AssertionError("orphaned turn must not be continued in this test")

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


class TestOrphanWatchdog:
    def test_no_continuation_within_orphan_timeout_interrupts_and_cancels(
        self, monkeypatch, client_module, session_module
    ):
        from dataclasses import dataclass

        @dataclass
        class ToolUseBlock:
            id: str
            name: str
            input: dict

        @dataclass
        class AssistantMessage:
            content: list
            usage: dict | None = None

        first_turn = [
            AssistantMessage(content=[ToolUseBlock(id="a", name="mcp__hermes__read_file", input={})])
        ]
        monkeypatch.setattr(
            client_module,
            "SdkSession",
            _fake_session_factory(first_turn=first_turn, session_module=session_module),
        )

        client = client_module.ClaudeSubClient(api_key="claude-sub", base_url="claude-sub://sdk")
        # Inject a short orphan_timeout without touching config.yaml.
        import dataclasses

        client._settings = dataclasses.replace(client._settings, orphan_timeout=0.2)

        tools = [
            {"type": "function", "function": {"name": "read_file", "description": "", "parameters": {}}}
        ]
        completion = client.chat.completions.create(
            model="claude-sonnet-5",
            messages=[{"role": "user", "content": "go"}],
            tools=tools,
            extra_body={"hermes_session_id": "sess-orphan"},
        )
        assert completion.choices[0].finish_reason == "tool_calls"

        turn = client._turns["sess-orphan"]
        pending_future = list(turn.pending.values())[0]
        timer = turn.orphan_timer
        assert timer is not None

        # Event-based wait: join the real watchdog timer thread (armed for
        # 0.2s) with generous slack, rather than a blind sleep.
        timer.join(timeout=2.0)
        assert not timer.is_alive()

        assert turn.session.interrupted is True
        assert pending_future.cancelled() is True
        assert "sess-orphan" not in client._turns

    def test_continuation_before_orphan_timeout_cancels_the_watchdog(
        self, monkeypatch, client_module, session_module
    ):
        from dataclasses import dataclass

        @dataclass
        class ToolUseBlock:
            id: str
            name: str
            input: dict

        @dataclass
        class TextBlock:
            text: str

        @dataclass
        class ResultMessage:
            is_error: bool = False
            result: str | None = None
            errors: list | None = None

        @dataclass
        class AssistantMessage:
            content: list
            usage: dict | None = None

        first_turn = [
            AssistantMessage(content=[ToolUseBlock(id="a", name="mcp__hermes__read_file", input={})])
        ]

        class _FakeSession:
            def __init__(self, **_kwargs):
                self.interrupted = False
                self.closed = False

            def run_turn(self, prompt, *, on_message, timeout=None, stall_timeout=None, stall_exempt=None):
                for message in first_turn:
                    try:
                        on_message(message)
                    except session_module.PauseTurn:
                        return len(first_turn)
                return len(first_turn)

            def continue_turn(self, *, on_message, timeout=None, stall_timeout=None, stall_exempt=None):
                for message in [AssistantMessage(content=[TextBlock(text="done")]), ResultMessage(result="done")]:
                    on_message(message)
                return 2

            def request_interrupt(self):
                self.interrupted = True
                return True

            def close(self):
                self.closed = True

        monkeypatch.setattr(client_module, "SdkSession", _FakeSession)
        client = client_module.ClaudeSubClient(api_key="claude-sub", base_url="claude-sub://sdk")
        import dataclasses

        client._settings = dataclasses.replace(client._settings, orphan_timeout=0.2)

        tools = [
            {"type": "function", "function": {"name": "read_file", "description": "", "parameters": {}}}
        ]
        first = client.chat.completions.create(
            model="claude-sonnet-5",
            messages=[{"role": "user", "content": "go"}],
            tools=tools,
            extra_body={"hermes_session_id": "sess-orphan-2"},
        )
        call = first.choices[0].message.tool_calls[0]
        timer = client._turns["sess-orphan-2"].orphan_timer
        assert timer is not None

        second = client.chat.completions.create(
            model="claude-sonnet-5",
            messages=[
                {"role": "user", "content": "go"},
                {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {"id": call.id, "type": "function", "function": {"name": "read_file", "arguments": "{}"}}
                    ],
                },
                {"role": "tool", "tool_call_id": call.id, "content": "ok"},
            ],
            tools=tools,
            extra_body={"hermes_session_id": "sess-orphan-2"},
        )
        assert second.choices[0].finish_reason == "stop"
        # cancel() just flags the timer; join deterministically before
        # asserting the thread actually exited (event-based, not a sleep).
        timer.join(timeout=2.0)
        assert not timer.is_alive()
