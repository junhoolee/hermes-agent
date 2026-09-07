"""``client.py`` bridge handler/projector id-binding reconciliation (D30-D34).

v0.1-D removed the earlier revision's polling wait for the bridge handler to
register before returning a tool-call response — a block whose bridge
handler is never actually invoked is not a bug case, it's just how the
CLI's own internal tool resolution (e.g. ``ToolSearch`` follow-ups)
normally behaves, and that wait turned every such block into a hard
timeout, a wrong continuation classification, and eventually a torn-down
CLI process. A ``ToolUseBlock`` is now turned into a ``tool_calls``
response the instant the projector sees it; the SDK's control-request
thread can invoke the matching bridge handler (``on_call``, via
``bridge.py``) before, during, or after that, or never at all, and binding
still has to succeed via ``(name, args)`` matching against whichever of
``turn.hook_seen`` / ``turn.outstanding`` / ``turn.unbound`` already has the
other half.

These tests exercise the real ``_note_tool_blocks``/``_make_on_call``
(client.py's own binding contract) directly, not through a scripted fake
SDK session — that layer (pump/classification/response shaping) is covered
by test_client_inversion.py/test_stream.py/test_watchdog.py, which now
install their own simple no-op autouse handler-binding fixture rather than
faking a wait that no longer exists.
"""

from __future__ import annotations

import queue
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
        assert turn.outstanding == {"toolu_1": ("read_file", client_module.args_key({}))}
        assert turn.handled_ids == {"toolu_1"}

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

    def test_two_calls_with_different_args_bind_via_exact_args_key_match(self, client_module):
        """A name match alone is ambiguous with two in-flight calls; args_key disambiguates."""
        turn = client_module._Turn(session=None, start_timeout=5.0)
        on_call = client_module._make_on_call(turn)

        fut_1 = on_call("read_file", {"path": "1"})
        fut_2 = on_call("read_file", {"path": "2"})

        projector = client_module._InversionProjector(turn=turn, expect_tools=True)
        message = AssistantMessage(
            content=[
                ToolUseBlock(id="a", name="mcp__hermes__read_file", input={"path": "2"}),
                ToolUseBlock(id="b", name="mcp__hermes__read_file", input={"path": "1"}),
            ]
        )
        with pytest.raises(client_module.PauseTurn):
            projector(message)

        assert turn.pending["a"] is fut_2
        assert turn.pending["b"] is fut_1


class TestProjectorFirst:
    """The block is projected — and sent to Hermes core — before ``on_call`` ever runs."""

    def test_tool_calls_returned_immediately_with_no_handler_ever_registered(self, client_module):
        """The core fix: no wait, no timeout — the block is never blocked on the handler."""
        turn = client_module._Turn(session=None, start_timeout=5.0)
        projector = client_module._InversionProjector(turn=turn, expect_tools=True)
        message = AssistantMessage(
            content=[ToolUseBlock(id="toolu_1", name="mcp__hermes__read_file", input={})]
        )

        started = time.monotonic()
        with pytest.raises(client_module.PauseTurn):
            projector(message)
        elapsed = time.monotonic() - started

        assert elapsed < 0.05
        assert projector.tool_calls == [("toolu_1", "read_file", "{}")]
        assert turn.pending == {}
        assert turn.outstanding == {"toolu_1": ("read_file", client_module.args_key({}))}

    def test_on_call_binds_via_outstanding_after_block_projected(self, client_module):
        turn = client_module._Turn(session=None, start_timeout=5.0)
        projector = client_module._InversionProjector(turn=turn, expect_tools=True)
        message = AssistantMessage(
            content=[ToolUseBlock(id="toolu_1", name="mcp__hermes__read_file", input={"path": "/x"})]
        )
        with pytest.raises(client_module.PauseTurn):
            projector(message)
        assert turn.pending == {}

        on_call = client_module._make_on_call(turn)
        fut = on_call("read_file", {"path": "/x"})

        assert turn.pending == {"toolu_1": fut}
        assert turn.handled_ids == {"toolu_1"}

    def test_two_parallel_calls_bind_via_args_key_in_either_arrival_order(self, client_module):
        turn = client_module._Turn(session=None, start_timeout=5.0)
        projector = client_module._InversionProjector(turn=turn, expect_tools=True)
        message = AssistantMessage(
            content=[
                ToolUseBlock(id="a", name="mcp__hermes__read_file", input={"path": "1"}),
                ToolUseBlock(id="b", name="mcp__hermes__read_file", input={"path": "2"}),
            ]
        )
        with pytest.raises(client_module.PauseTurn):
            projector(message)

        on_call = client_module._make_on_call(turn)
        # Handlers dispatch in reverse of block order — still binds correctly.
        fut_b = on_call("read_file", {"path": "2"})
        fut_a = on_call("read_file", {"path": "1"})

        assert turn.pending == {"a": fut_a, "b": fut_b}


class TestHookSeenBinding:
    """``PreToolUse`` hook observation (``on_pre_tool_use``) beats out ``outstanding``."""

    def test_hook_seen_before_handler_and_before_block(self, client_module):
        turn = client_module._Turn(session=None, start_timeout=5.0)
        on_pre_tool_use = client_module._make_on_pre_tool_use(turn)
        on_pre_tool_use("toolu_1", "read_file", {"path": "/x"})
        assert turn.hook_seen == [("toolu_1", "read_file", client_module.args_key({"path": "/x"}))]

        on_call = client_module._make_on_call(turn)
        fut = on_call("read_file", {"path": "/x"})
        assert turn.hook_seen == []
        assert turn.pending == {"toolu_1": fut}
        assert turn.handled_ids == {"toolu_1"}

        # The block, when it eventually arrives with the SAME id as the hook
        # (the real-world case — the hook and the block are keyed off the
        # same underlying API tool_use id), finds the Future already in
        # pending via the outstanding-registration path in _note_tool_blocks
        # and leaves it alone (nothing left in unbound to steal).
        projector = client_module._InversionProjector(turn=turn, expect_tools=True)
        message = AssistantMessage(
            content=[ToolUseBlock(id="toolu_1", name="mcp__hermes__read_file", input={"path": "/x"})]
        )
        with pytest.raises(client_module.PauseTurn):
            projector(message)
        assert turn.pending == {"toolu_1": fut}


class TestStashedContinuationResult:
    """Continuation result arrives before the handler ever calls ``on_call``."""

    def test_result_delivered_before_handler_is_stashed_then_bound(self, client_module):
        turn = client_module._Turn(session=None, start_timeout=5.0)
        projector = client_module._InversionProjector(turn=turn, expect_tools=True)
        message = AssistantMessage(
            content=[ToolUseBlock(id="toolu_1", name="mcp__hermes__read_file", input={})]
        )
        with pytest.raises(client_module.PauseTurn):
            projector(message)
        assert turn.outstanding == {"toolu_1": ("read_file", client_module.args_key({}))}

        # Simulate the continuation delivering the tool result before any
        # handler ever ran (Hermes core resolved the call some other way).
        payload = {"content": [{"type": "text", "text": "hello"}], "is_error": False}
        with turn.lock:
            turn.results["toolu_1"] = payload

        on_call = client_module._make_on_call(turn)
        fut = on_call("read_file", {})

        assert fut.done()
        assert fut.result() == payload
        assert "toolu_1" not in turn.results
        assert turn.handled_ids == {"toolu_1"}


class TestNeverHandledBlockCompletesNormally:
    """No handler is ever invoked for a sent block — the turn still completes."""

    def test_resolve_pending_stashes_and_clears_outstanding_without_error(self, client_module):
        turn = client_module._Turn(session=None, start_timeout=5.0)
        projector = client_module._InversionProjector(turn=turn, expect_tools=True)
        message = AssistantMessage(
            content=[ToolUseBlock(id="toolu_1", name="mcp__hermes__read_file", input={})]
        )
        with pytest.raises(client_module.PauseTurn):
            projector(message)

        class _FakeClient:
            def _resolve_pending(self, turn, tail):
                client_module.ClaudeSubClient._resolve_pending(self, turn, tail)

        fake = _FakeClient()
        tail = [{"role": "tool", "tool_call_id": "toolu_1", "content": "ok"}]
        fake._resolve_pending(turn, tail)

        assert turn.outstanding == {}
        assert turn.pending == {}
        assert turn.results == {
            "toolu_1": {"content": [{"type": "text", "text": "ok"}], "is_error": False}
        }


class TestCliResolvedBlocksAreNotSent:
    """A block already resolved by the CLI itself (PostToolUse, no handler) is dropped."""

    def test_note_tool_blocks_skips_cli_resolved_ids(self, client_module):
        turn = client_module._Turn(session=None, start_timeout=5.0)
        turn.cli_resolved.add("toolu_1")
        block = ToolUseBlock(id="toolu_1", name="mcp__hermes__read_file", input={})

        sendable = client_module._note_tool_blocks(turn, [block])

        assert sendable == []
        assert turn.outstanding == {}
        assert turn.pending == {}

    def test_projector_does_not_pause_when_all_blocks_are_cli_resolved(self, client_module):
        turn = client_module._Turn(session=None, start_timeout=5.0)
        turn.cli_resolved.add("toolu_1")
        projector = client_module._InversionProjector(turn=turn, expect_tools=True)
        message = AssistantMessage(
            content=[ToolUseBlock(id="toolu_1", name="mcp__hermes__read_file", input={})]
        )
        # No PauseTurn — the projector keeps draining since nothing is sendable.
        projector(message)
        assert projector.tool_calls is None

    def test_post_tool_use_marks_cli_resolved_only_when_not_handled(self, client_module):
        turn = client_module._Turn(session=None, start_timeout=5.0)
        on_post_tool_use = client_module._make_on_post_tool_use(turn)

        on_post_tool_use("toolu_1", "read_file", {"content": []}, False)
        assert turn.cli_resolved == {"toolu_1"}

        turn.handled_ids.add("toolu_2")
        on_post_tool_use("toolu_2", "read_file", {"content": []}, False)
        assert turn.cli_resolved == {"toolu_1"}


class TestAbortCancelsPendingAndUnbound:
    def test_abort_turn_cancels_and_clears_everything(self, client_module):
        class _FakeSession:
            def __init__(self):
                self.interrupted = False
                self.closed = False

            def request_interrupt(self):
                self.interrupted = True
                return True

            def close(self):
                self.closed = True

        turn = client_module._Turn(session=_FakeSession(), start_timeout=5.0)
        on_call = client_module._make_on_call(turn)
        pending_fut = on_call("read_file", {})
        pending_fut_2 = None
        # Bind pending_fut via a matching block so it lands in turn.pending.
        _note = client_module._note_tool_blocks(
            turn, [ToolUseBlock(id="toolu_1", name="mcp__hermes__read_file", input={})]
        )
        assert turn.pending == {"toolu_1": pending_fut}
        unbound_fut = on_call("terminal", {"command": "ls"})
        assert turn.unbound == [("terminal", client_module.args_key({"command": "ls"}), unbound_fut)]

        client = client_module.ClaudeSubClient(api_key="claude-sub", base_url="claude-sub://sdk")
        client._abort_turn(turn)

        assert pending_fut.cancelled()
        assert unbound_fut.cancelled()
        assert turn.pending == {}
        assert turn.outstanding == {}
        assert turn.hook_seen == []
        assert turn.unbound == []
        assert turn.results == {}
        assert turn.handled_ids == set()
        assert turn.cli_resolved == set()
        assert turn.session.interrupted is True
        assert turn.session.closed is True
