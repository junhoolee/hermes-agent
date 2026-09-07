"""``client.py`` bridge handler/projector id-binding race (D21).

The SDK's control-request thread can invoke the bridge handler
(``on_call``, via ``bridge.py``) either before or after the drain thread
projects the matching ``AssistantMessage`` tool-use block — there is no
ordering guarantee between the two (see ``client.py``'s module docstring
and the ``_wait_for_pending`` docstring). These tests exercise the real
``_wait_for_pending`` (only ``BRIDGE_BIND_TIMEOUT`` is ever monkeypatched,
and only for the timeout case) against both arrival orders, instead of
replacing it with a fake like ``test_client_inversion.py`` and
``test_watchdog.py`` do — those tests cover client.py's inversion contract
assuming binding already happened; this file covers the binding itself.
"""

from __future__ import annotations

import queue
import threading
import time
from dataclasses import dataclass

import pytest


@dataclass
class ToolUseBlock:
    id: str
    name: str
    input: dict


@dataclass
class AssistantMessage:
    content: list
    usage: dict | None = None


READ_FILE_TOOL = [
    {
        "type": "function",
        "function": {"name": "read_file", "description": "Read a file.", "parameters": {}},
    }
]


@pytest.fixture
def client_module(load_plugin_module):
    return load_plugin_module("client")


@pytest.fixture
def session_module(load_plugin_module):
    return load_plugin_module("session")


class TestHandlerFirst:
    """``on_call`` parks in ``turn.unbound`` before the block is ever seen."""

    def test_single_call_binds_via_unbound_park(self, client_module):
        turn = client_module._Turn(session=None, start_timeout=5.0)
        on_call = client_module._make_on_call(turn)

        fut = on_call("read_file", {})

        projector = client_module._InversionProjector(turn=turn, expect_tools=True)
        message = AssistantMessage(
            content=[ToolUseBlock(id="toolu_1", name="mcp__hermes__read_file", input={})]
        )
        with pytest.raises(client_module.PauseTurn):
            projector(message)

        assert turn.pending == {"toolu_1": fut}
        assert turn.pending["toolu_1"] is fut
        assert turn.unbound == []
        assert turn.expected_ids == []

    def test_two_parallel_calls_bind_in_registration_order(self, client_module):
        turn = client_module._Turn(session=None, start_timeout=5.0)
        on_call = client_module._make_on_call(turn)

        fut_a = on_call("read_file", {})
        fut_b = on_call("read_file", {})

        projector = client_module._InversionProjector(turn=turn, expect_tools=True)
        message = AssistantMessage(
            content=[
                ToolUseBlock(id="a", name="mcp__hermes__read_file", input={}),
                ToolUseBlock(id="b", name="mcp__hermes__read_file", input={}),
            ]
        )
        with pytest.raises(client_module.PauseTurn):
            projector(message)

        assert turn.pending == {"a": fut_a, "b": fut_b}
        assert turn.unbound == []
        assert turn.expected_ids == []

    def test_streaming_projector_also_binds_via_unbound_park(self, client_module):
        turn = client_module._Turn(session=None, start_timeout=5.0)
        on_call = client_module._make_on_call(turn)

        fut = on_call("read_file", {})

        projector = client_module._StreamingProjector(
            turn=turn, expect_tools=True, chunk_queue=queue.Queue(), model="claude-sonnet-5"
        )
        message = AssistantMessage(
            content=[ToolUseBlock(id="toolu_1", name="mcp__hermes__read_file", input={})]
        )
        with pytest.raises(client_module.PauseTurn):
            projector(message)

        assert turn.pending == {"toolu_1": fut}


class TestProjectorFirst:
    """The block is projected (and ``_wait_for_pending`` starts polling) before ``on_call`` runs."""

    def test_projector_blocks_until_handler_registers(self, client_module):
        turn = client_module._Turn(session=None, start_timeout=5.0)
        on_call = client_module._make_on_call(turn)
        paused = threading.Event()

        def _run_projector() -> None:
            projector = client_module._InversionProjector(turn=turn, expect_tools=True)
            message = AssistantMessage(
                content=[ToolUseBlock(id="toolu_1", name="mcp__hermes__read_file", input={})]
            )
            try:
                projector(message)
            except client_module.PauseTurn:
                paused.set()

        thread = threading.Thread(target=_run_projector, daemon=True)
        thread.start()
        time.sleep(0.05)

        fut = on_call("read_file", {})

        thread.join(timeout=2.0)
        assert not thread.is_alive()
        assert paused.is_set()
        assert turn.pending == {"toolu_1": fut}


class TestBindingTimeout:
    """No matching ``on_call`` ever arrives — the wait must not hang forever."""

    def test_wait_for_pending_raises_bridge_bind_timeout(self, monkeypatch, client_module):
        monkeypatch.setattr(client_module, "BRIDGE_BIND_TIMEOUT", 0.3)
        turn = client_module._Turn(session=None, start_timeout=5.0)

        started = time.monotonic()
        with pytest.raises(client_module.BridgeBindTimeout):
            client_module._wait_for_pending(turn, ["toolu_never_bound"])
        assert time.monotonic() - started < 2.0

    def test_unbound_tool_call_maps_to_503_and_discards_turn(
        self, monkeypatch, client_module, session_module, fake_sdk
    ):
        monkeypatch.setattr(client_module, "BRIDGE_BIND_TIMEOUT", 0.3)
        closed_flag = {"closed": False}

        class _FakeSession:
            def __init__(self, **_kwargs) -> None:
                self.closed = False

            def run_turn(self, prompt, *, on_message, timeout=None, stall_timeout=None, stall_exempt=None):
                # No bridge handler ever calls on_call for this block —
                # simulates a handler that never fires (wedged/crashed).
                on_message(
                    AssistantMessage(
                        content=[
                            ToolUseBlock(
                                id="toolu_1", name="mcp__hermes__read_file", input={"path": "/tmp/x"}
                            )
                        ]
                    )
                )
                return 1

            def continue_turn(self, **_kwargs):
                raise AssertionError("continue_turn should not be reached")

            def request_interrupt(self) -> bool:
                return True

            def close(self) -> None:
                self.closed = True
                closed_flag["closed"] = True

        monkeypatch.setattr(client_module, "SdkSession", lambda **_kwargs: _FakeSession())
        client = client_module.ClaudeSubClient(api_key="claude-sub", base_url="claude-sub://sdk")

        import openai

        with pytest.raises(openai.APIStatusError) as excinfo:
            client.chat.completions.create(
                model="claude-sonnet-5",
                messages=[{"role": "user", "content": "read /tmp/x"}],
                tools=READ_FILE_TOOL,
                extra_body={"hermes_session_id": "sess-timeout"},
            )

        assert excinfo.value.status_code == 503
        assert "sess-timeout" not in client._turns
        assert closed_flag["closed"] is True
