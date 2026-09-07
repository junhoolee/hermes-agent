"""``SdkSession`` — the synchronous facade over the async claude-agent-sdk client.

Exercises the real event-loop-thread bridge (``ensure_started``, ``run_turn``,
``request_interrupt``, ``close``) against an injected fake async client, so
these tests pin real threading/asyncio behavior (ordering, timeouts, stalls)
rather than mocking it away.
"""

from __future__ import annotations

import asyncio

import pytest


class _FakeAsyncClient:
    """Minimal async stand-in for ``ClaudeSDKClient``."""

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
    """Returns (factory, holder) — holder[0] is the constructed client."""
    holder: list[_FakeAsyncClient] = []

    def factory(*, options):
        client = _FakeAsyncClient(options=options)
        holder.append(client)
        return client

    return factory, holder


def _session(load_plugin_module, factory, *, start_timeout=5.0, close_timeout=5.0):
    session_mod = load_plugin_module("session")
    return session_mod.SdkSession(
        options_factory=lambda: object(),
        client_factory=factory,
        transport_factory=None,
        start_timeout=start_timeout,
        close_timeout=close_timeout,
    )


class TestLifecycle:
    def test_ensure_started_connects_client(self, load_plugin_module, fake_client_factory):
        factory, holder = fake_client_factory
        session = _session(load_plugin_module, factory)
        session.ensure_started()
        try:
            assert session.started is True
            assert holder[0].connected is True
        finally:
            session.close()

    def test_ensure_started_is_idempotent(self, load_plugin_module, fake_client_factory):
        factory, holder = fake_client_factory
        session = _session(load_plugin_module, factory)
        session.ensure_started()
        session.ensure_started()
        try:
            assert len(holder) == 1
        finally:
            session.close()

    def test_close_disconnects_and_is_idempotent(self, load_plugin_module, fake_client_factory):
        factory, holder = fake_client_factory
        session = _session(load_plugin_module, factory)
        session.ensure_started()
        session.close()
        assert holder[0].connected is False
        assert session.closed is True
        session.close()  # must not raise

    def test_ensure_started_after_close_raises(self, load_plugin_module, fake_client_factory):
        session_mod = load_plugin_module("session")
        factory, _holder = fake_client_factory
        session = _session(load_plugin_module, factory)
        session.ensure_started()
        session.close()
        with pytest.raises(session_mod.SdkSessionError):
            session.ensure_started()


class TestRunTurn:
    def test_delivers_messages_in_order_and_returns_count(self, load_plugin_module, fake_client_factory):
        factory, holder = fake_client_factory
        session = _session(load_plugin_module, factory)
        session.ensure_started()
        holder[0].messages_factory = lambda: ["m1", "m2"]
        received: list[str] = []
        try:
            delivered = session.run_turn("hello", on_message=received.append, timeout=5.0)
            assert delivered == 2
            assert received == ["m1", "m2"]
            assert holder[0].queries == ["hello"]
        finally:
            session.close()

    def test_stops_draining_shortly_after_result_message(self, load_plugin_module, fake_client_factory):
        session_mod = load_plugin_module("session")

        class _Result:
            pass

        _Result.__name__ = "ResultMessage"

        factory, holder = fake_client_factory
        session = _session(load_plugin_module, factory)
        session.ensure_started()
        holder[0].messages_factory = lambda: ["m1", _Result()]
        received: list = []
        try:
            delivered = session.run_turn("hello", on_message=received.append, timeout=5.0)
            assert delivered == 2
            assert session_mod.is_result_message(received[-1])
        finally:
            session.close()

    def test_stall_timeout_raises_timeout_error(self, load_plugin_module, fake_client_factory):
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
        finally:
            session.close()

    def test_turn_timeout_raises_timeout_error(self, load_plugin_module, fake_client_factory):
        factory, holder = fake_client_factory
        session = _session(load_plugin_module, factory)
        session.ensure_started()
        holder[0].messages_factory = lambda: [f"m{i}" for i in range(50)]
        holder[0]._message_delay = 0.05  # keeps arriving, so stall never trips
        try:
            with pytest.raises(TimeoutError):
                session.run_turn(
                    "hello", on_message=lambda _m: None, timeout=0.3, stall_timeout=None
                )
        finally:
            session.close()


class TestInterrupt:
    def test_request_interrupt_calls_client(self, load_plugin_module, fake_client_factory):
        factory, holder = fake_client_factory
        session = _session(load_plugin_module, factory)
        session.ensure_started()
        try:
            assert session.request_interrupt() is True
            assert holder[0].interrupted is True
        finally:
            session.close()

    def test_request_interrupt_before_start_returns_false(self, load_plugin_module, fake_client_factory):
        factory, _holder = fake_client_factory
        session = _session(load_plugin_module, factory)
        assert session.request_interrupt() is False


class TestBuildOptions:
    def test_builds_tool_less_single_turn_options(self, load_plugin_module, fake_sdk):
        session_mod = load_plugin_module("session")
        options = session_mod.build_options(model="claude-sonnet-5", cwd="/work")
        assert options.allowed_tools == []
        assert options.tools == []
        assert options.max_turns == 1
        assert options.mcp_servers == {}
        assert options.cwd == "/work"
        assert options.model == "claude-sonnet-5"
        assert options.system_prompt["preset"] == "claude_code"
        assert options.system_prompt["append"] == session_mod.DEFAULT_IDENTITY_APPEND

    def test_identity_append_override(self, load_plugin_module, fake_sdk):
        session_mod = load_plugin_module("session")
        options = session_mod.build_options(identity_append="custom identity", cwd="/work")
        assert options.system_prompt["append"] == "custom identity"

    def test_valid_reasoning_effort_is_forwarded(self, load_plugin_module, fake_sdk):
        session_mod = load_plugin_module("session")
        options = session_mod.build_options(reasoning_effort="high", cwd="/work")
        assert options.effort == "high"

    def test_invalid_reasoning_effort_is_dropped(self, load_plugin_module, fake_sdk):
        session_mod = load_plugin_module("session")
        options = session_mod.build_options(reasoning_effort="bogus", cwd="/work")
        assert options.effort is None
