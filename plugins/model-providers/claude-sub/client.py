"""OpenAI-client-shaped facade over the claude-agent-sdk for claude-sub.

Hermes core owns the tool-calling loop; the SDK's own turn only ever runs
until the *next* tool call. ``create()`` inverts control between the two:

* A **new turn** (a fresh user message, or the first call for a session)
  opens an ``SdkSession`` and runs it until either a ``ResultMessage``
  (turn done, ``finish_reason="stop"``) or an ``AssistantMessage`` carrying
  ``mcp__hermes__*`` tool-use blocks — at that point the SDK turn is left
  *open* (paused, not torn down) and a synthetic ``tool_calls`` response is
  returned so Hermes core can run those tools itself.
* A **continuation** (Hermes core's follow-up ``create()`` carrying the
  ``tool`` result messages for exactly the pending call ids) resolves the
  bridge's waiting Futures and resumes draining the *same* SDK turn — no
  new ``query()`` — until the next pause or the final ``ResultMessage``.

v0.1-D: a ``ToolUseBlock`` is turned into a ``tool_calls`` response the
*instant* the projector sees it — no polling, no timeout, no wait of any
kind. The SDK's own control channel invokes the matching bridge handler
(``on_call``, via ``bridge.py``) on a separate thread with no ordering
guarantee relative to the projector seeing the block; binding the two
together (and the eventual tool result) is handled entirely by matching on
``(name, args)`` — via the ``PreToolUse``/``PostToolUse`` hooks
(``bridge.build_hooks``), via whichever of ``on_call``/the block arrives
second consulting whichever arrived first, or, if the block is sent to
Hermes core before the handler ever runs, by stashing the eventual
continuation result until the handler catches up. See ``_note_tool_blocks``
and ``_make_on_call`` below for the full reconciliation, and ``session.py``
for the pause/continue primitive this relies on (``PauseTurn`` /
``continue_turn``).

Earlier revisions of this module (see git history for v0.1-C and prior) had
the projector block on a short poll for the handler to register before
returning — that design was fundamentally unsound (a bridge handler for a
block that is never actually invoked is not a bug case, it's how the CLI's
own ``ToolSearch``-driven internal tool resolution normally behaves — see
``bridge.PASSTHROUGH_TOOLS`` and D34's ``cli_resolved``) and is gone here,
not just retried with a longer timeout.
"""

from __future__ import annotations

import concurrent.futures
import hashlib
import json
import logging
import os
import queue
import threading
import time
from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import Any

from . import bridge, convert, errors
from .bridge import BRIDGE_PREFIX
from .config import load_settings
from .session import PauseTurn, SdkSession, build_options

logger = logging.getLogger(__name__)

MARKER_BASE_URL = "claude-sub://sdk"

_STREAM_DONE = object()


def args_key(args: dict | None) -> str:
    """Stable string key for matching a handler call against a tool-use block."""
    return json.dumps(args or {}, sort_keys=True, ensure_ascii=False, default=str)


@dataclass
class _Turn:
    """One open Hermes<->SDK turn, keyed by session_key in ``ClaudeSubClient``.

    ``state`` tracks where this turn sits in its lifecycle: ``"open"`` while
    a drain (``run_turn``/``continue_turn``) is in flight, ``"paused"`` after
    a ``tool_calls`` response has been handed back and Hermes core's
    continuation is awaited, and ``"idle"`` after a ``stop`` when the
    underlying ``SdkSession`` (and its CLI subprocess) is being kept warm for
    a possible follow-up turn on the same conversation (v0.1-E). The
    ``model``/``reasoning_effort``/``tools_key``/``system_hash`` fields pin
    the parameters an idle session was opened with so a warm follow-up can be
    refused the moment any of them changes; ``seen_count``/``last_reply_text``
    let ``_is_warm_followup`` verify Hermes core's history still matches what
    this session actually said before trusting a warm reuse.
    """

    session: Any
    start_timeout: float
    pending: dict = field(default_factory=dict)  # call_id -> concurrent.futures.Future
    outstanding: dict = field(default_factory=dict)  # call_id -> (name, args_key)
    hook_seen: list = field(default_factory=list)  # [(tool_use_id, name, args_key), ...]
    unbound: list = field(default_factory=list)  # [(name, args_key, Future), ...]
    results: dict = field(default_factory=dict)  # call_id -> stashed tool-result payload
    handled_ids: set = field(default_factory=set)  # call ids an actual handler ran for
    cli_resolved: set = field(default_factory=set)  # call ids the CLI resolved itself
    lock: threading.RLock = field(default_factory=threading.RLock)
    opened_at: float = field(default_factory=time.monotonic)
    orphan_timer: Any = None
    state: str = "open"
    model: str | None = None
    reasoning_effort: str | None = None
    tools_key: str | None = None
    system_hash: str | None = None
    seen_count: int = 0
    last_reply_text: str = ""
    idle_timer: Any = None
    has_session_id: bool = False


def _derive_session_key(system_text: str, first_user_text: str) -> str:
    digest = hashlib.sha1(f"{system_text}\x00{first_user_text}".encode()).hexdigest()
    return digest[:16]


def _tools_key(tools: list[dict] | None) -> str:
    """Stable key for the tool set a turn was opened with (order-independent)."""
    if not tools:
        return "none"
    names = sorted(
        (tool.get("function") or {}).get("name", "") for tool in tools if isinstance(tool, dict)
    )
    return hashlib.sha1("\x00".join(names).encode()).hexdigest()


def _system_hash(system_text: str) -> str:
    return hashlib.sha1((system_text or "").encode()).hexdigest()


def _is_warm_followup(
    messages: list[dict],
    turn: "_Turn",
    *,
    model: str | None,
    reasoning_effort: str | None,
    tools: list[dict] | None,
    extra_body: dict | None,
) -> bool:
    """True when *messages* is exactly this idle turn's history plus new user text.

    Every check below must pass — any mismatch is treated as a cold start
    (safety over reuse). See ``_Turn`` and D35-5 in the v0.1-E design notes.
    """
    if not (isinstance(extra_body, dict) and extra_body.get("hermes_session_id")):
        logger.info("claude-sub: warm=False reason=no_session_id")
        return False
    if len(messages) <= turn.seen_count:
        logger.info("claude-sub: warm=False reason=no_new_messages")
        return False
    prev = messages[turn.seen_count - 1]
    if not isinstance(prev, dict) or prev.get("role") != "assistant":
        logger.info("claude-sub: warm=False reason=history_mismatch")
        return False
    if convert.text_from_content(prev.get("content")).strip() != turn.last_reply_text:
        logger.info("claude-sub: warm=False reason=history_mismatch")
        return False
    tail = messages[turn.seen_count :]
    if not tail or any(
        not isinstance(message, dict) or message.get("role") != "user" for message in tail
    ):
        logger.info("claude-sub: warm=False reason=history_mismatch")
        return False
    system_text, _last_user_text, _prior = convert.split_messages(messages)
    if _system_hash(system_text) != turn.system_hash:
        logger.info("claude-sub: warm=False reason=system_changed")
        return False
    if model != turn.model:
        logger.info("claude-sub: warm=False reason=model_changed")
        return False
    if reasoning_effort != turn.reasoning_effort:
        logger.info("claude-sub: warm=False reason=reasoning_changed")
        return False
    if _tools_key(tools) != turn.tools_key:
        logger.info("claude-sub: warm=False reason=tools_changed")
        return False
    return True


def _note_tool_blocks(turn: _Turn, tool_blocks: list) -> list:
    """Register each tool-use *block* and return the subset to send to Hermes core.

    Called the instant the projector sees the blocks — no waiting. If a
    bridge handler already parked a Future in ``turn.unbound`` for this
    block's ``(name, args)`` (the handler-arrived-first race), claim it
    into ``pending`` right away; either way the block's id is recorded in
    ``turn.outstanding`` so a same-thread-later ``on_call`` invocation (the
    far more common block-arrived-first race, since the SDK's control
    channel only starts a handler once it has fully dispatched the tool
    call) can find it.

    A block whose id is already in ``turn.cli_resolved`` (the CLI itself
    resolved that call — see ``bridge.build_hooks``'s ``PostToolUse``
    wiring — meaning no handler will ever run for it, e.g. an internal
    ``ToolSearch`` follow-up) is dropped: it is never sent to Hermes core.
    """
    sendable = []
    with turn.lock:
        for block in tool_blocks:
            name = getattr(block, "name", "") or ""
            short_name = name[len(BRIDGE_PREFIX) :] if name.startswith(BRIDGE_PREFIX) else name
            call_id = getattr(block, "id", None)
            if call_id in turn.cli_resolved:
                logger.info(
                    "claude-sub: tool_use block %s (%s) already resolved by the CLI itself; "
                    "not sending to Hermes core",
                    call_id,
                    short_name,
                )
                continue
            key = args_key(getattr(block, "input", None))
            fut = None
            for index, (u_name, u_key, u_fut) in enumerate(turn.unbound):
                if u_name == short_name and u_key == key:
                    fut = u_fut
                    del turn.unbound[index]
                    break
            if fut is None:
                for index, (u_name, _u_key, u_fut) in enumerate(turn.unbound):
                    if u_name == short_name:
                        fut = u_fut
                        del turn.unbound[index]
                        break
            if fut is not None:
                turn.pending[call_id] = fut
                turn.handled_ids.add(call_id)
            turn.outstanding[call_id] = (short_name, key)
            sendable.append(block)
    return sendable


def _make_on_call(turn: _Turn):
    def on_call(name: str, args: dict) -> "concurrent.futures.Future":
        key = args_key(args)
        fut: "concurrent.futures.Future" = concurrent.futures.Future()
        with turn.lock:
            call_id = None
            for index, (tool_use_id, hook_name, hook_key) in enumerate(turn.hook_seen):
                if hook_name == name and hook_key == key:
                    call_id = tool_use_id
                    del turn.hook_seen[index]
                    break
            if call_id is None:
                candidates = [
                    (cid, o_name, o_key)
                    for cid, (o_name, o_key) in turn.outstanding.items()
                    if cid not in turn.pending and cid not in turn.handled_ids
                ]
                for cid, o_name, o_key in candidates:
                    if o_name == name and o_key == key:
                        call_id = cid
                        break
                if call_id is None:
                    for cid, o_name, _o_key in candidates:
                        if o_name == name:
                            call_id = cid
                            break
            if call_id is not None:
                turn.handled_ids.add(call_id)
                stashed = turn.results.pop(call_id, None)
                if stashed is not None:
                    logger.info(
                        "claude-sub: bridge handler for %s (%s) found a stashed continuation "
                        "result; resolving immediately",
                        name,
                        call_id,
                    )
                    fut.set_result(stashed)
                else:
                    turn.pending[call_id] = fut
            else:
                turn.unbound.append((name, key, fut))
                logger.debug(
                    "claude-sub: bridge handler for %s arrived unbound (no matching hook_seen "
                    "or outstanding entry yet); parking",
                    name,
                )
        return fut

    return on_call


def _classify_continuation(messages: list[dict], turn: "_Turn | None") -> list[dict] | None:
    """Return the trailing tool-result messages if *messages* continues *turn*, else None."""
    if turn is None:
        return None
    with turn.lock:
        outstanding_ids = set(turn.outstanding.keys())
    if not outstanding_ids:
        return None

    last_assistant_idx = None
    for index in range(len(messages) - 1, -1, -1):
        message = messages[index]
        if isinstance(message, dict) and message.get("role") == "assistant":
            last_assistant_idx = index
            break
    if last_assistant_idx is None:
        return None

    assistant_msg = messages[last_assistant_idx]
    tool_calls = assistant_msg.get("tool_calls") or []
    assistant_ids = {tc.get("id") for tc in tool_calls if isinstance(tc, dict) and tc.get("id")}
    if not assistant_ids:
        return None

    tail = messages[last_assistant_idx + 1 :]
    if not tail:
        return None
    tail_ids = set()
    for message in tail:
        if not isinstance(message, dict) or message.get("role") != "tool":
            return None
        call_id = message.get("tool_call_id")
        if not call_id:
            return None
        tail_ids.add(call_id)

    if tail_ids != assistant_ids or tail_ids != outstanding_ids:
        return None
    return tail


class _InversionProjector:
    """Accumulates one turn's text/thinking/usage and detects tool-call pauses."""

    def __init__(self, *, turn: _Turn, expect_tools: bool) -> None:
        self.turn = turn
        self.expect_tools = expect_tools
        self.text_parts: list[str] = []
        self.thinking_parts: list[str] = []
        self.last_call_usage: dict | None = None
        self.result_message: Any = None
        self.tool_calls: list[tuple[str, str, str]] | None = None

    def __call__(self, message: Any) -> None:
        kind = type(message).__name__
        if kind == "AssistantMessage":
            usage = getattr(message, "usage", None)
            if isinstance(usage, dict) and usage:
                self.last_call_usage = usage
            tool_blocks = []
            for block in getattr(message, "content", None) or []:
                block_kind = type(block).__name__
                if block_kind == "TextBlock":
                    text = getattr(block, "text", "") or ""
                    if text:
                        self.text_parts.append(text)
                elif block_kind == "ThinkingBlock":
                    thinking = getattr(block, "thinking", "") or ""
                    if thinking:
                        self.thinking_parts.append(thinking)
                elif block_kind == "ToolUseBlock" and self.expect_tools:
                    name = getattr(block, "name", "") or ""
                    if name.startswith(BRIDGE_PREFIX):
                        tool_blocks.append(block)
            if tool_blocks:
                sendable = _note_tool_blocks(self.turn, tool_blocks)
                if sendable:
                    self.tool_calls = [
                        (
                            getattr(block, "id", None),
                            getattr(block, "name", "")[len(BRIDGE_PREFIX) :],
                            json.dumps(getattr(block, "input", None) or {}, ensure_ascii=False),
                        )
                        for block in sendable
                    ]
                    logger.info(
                        "claude-sub: tool_calls ready ids=%s",
                        [call_id for call_id, _name, _args in self.tool_calls],
                    )
                    raise PauseTurn()
        elif kind == "ResultMessage":
            self.result_message = message


def _usage_namespace(usage_dict: dict) -> SimpleNamespace:
    return SimpleNamespace(
        prompt_tokens=usage_dict["prompt_tokens"],
        completion_tokens=usage_dict["completion_tokens"],
        total_tokens=usage_dict["total_tokens"],
        prompt_tokens_details=SimpleNamespace(cached_tokens=usage_dict["cached_tokens"]),
    )


def _stream_delta(*, content=None, tool_calls=None, reasoning_content=None) -> SimpleNamespace:
    return SimpleNamespace(
        role="assistant",
        content=content,
        tool_calls=tool_calls,
        reasoning_content=reasoning_content,
        reasoning=reasoning_content,
    )


def _data_chunk(model: str, *, content=None, reasoning_content=None, finish_reason=None):
    return SimpleNamespace(
        choices=[
            SimpleNamespace(
                index=0,
                delta=_stream_delta(content=content, reasoning_content=reasoning_content),
                finish_reason=finish_reason,
            )
        ],
        model=model,
        usage=None,
    )


def _tool_calls_delta_chunk(model: str, tool_calls: list[tuple[str, str, str]]):
    deltas = [
        SimpleNamespace(
            index=index,
            id=call_id,
            type="function",
            function=SimpleNamespace(name=name, arguments=arguments),
        )
        for index, (call_id, name, arguments) in enumerate(tool_calls)
    ]
    return SimpleNamespace(
        choices=[
            SimpleNamespace(
                index=0,
                delta=_stream_delta(tool_calls=deltas),
                finish_reason="tool_calls",
            )
        ],
        model=model,
        usage=None,
    )


def _stop_chunk(model: str):
    return SimpleNamespace(
        choices=[SimpleNamespace(index=0, delta=_stream_delta(), finish_reason="stop")],
        model=model,
        usage=None,
    )


def _usage_chunk(model: str, last_call_usage: dict | None):
    usage_dict = convert.usage_from_assistant(last_call_usage)
    return SimpleNamespace(choices=[], model=model, usage=_usage_namespace(usage_dict))


class _StreamingProjector:
    """Like ``_InversionProjector``, but pushes OpenAI-shaped delta chunks live."""

    def __init__(self, *, turn: _Turn, expect_tools: bool, chunk_queue, model: str) -> None:
        self.turn = turn
        self.expect_tools = expect_tools
        self.chunk_queue = chunk_queue
        self.model = model
        self.last_call_usage: dict | None = None
        self.result_message: Any = None
        self.tool_calls: list[tuple[str, str, str]] | None = None
        self.text_parts: list[str] = []
        self._streamed_text = False

    def __call__(self, message: Any) -> None:
        kind = type(message).__name__
        if kind == "StreamEvent":
            event = getattr(message, "event", None)
            if not isinstance(event, dict) or event.get("type") != "content_block_delta":
                return
            delta = event.get("delta") or {}
            delta_type = delta.get("type")
            if delta_type == "text_delta":
                text = delta.get("text") or ""
                if text:
                    self._streamed_text = True
                    self.text_parts.append(text)
                    self.chunk_queue.put(_data_chunk(self.model, content=text))
            elif delta_type == "thinking_delta":
                thinking = delta.get("thinking") or ""
                if thinking:
                    self.chunk_queue.put(_data_chunk(self.model, reasoning_content=thinking))
            return
        if kind == "AssistantMessage":
            usage = getattr(message, "usage", None)
            if isinstance(usage, dict) and usage:
                self.last_call_usage = usage
            tool_blocks = []
            for block in getattr(message, "content", None) or []:
                block_kind = type(block).__name__
                if block_kind == "TextBlock":
                    if not self._streamed_text:
                        text = getattr(block, "text", "") or ""
                        if text:
                            self.text_parts.append(text)
                            self.chunk_queue.put(_data_chunk(self.model, content=text))
                elif block_kind == "ToolUseBlock" and self.expect_tools:
                    name = getattr(block, "name", "") or ""
                    if name.startswith(BRIDGE_PREFIX):
                        tool_blocks.append(block)
            if tool_blocks:
                sendable = _note_tool_blocks(self.turn, tool_blocks)
                if sendable:
                    self.tool_calls = [
                        (
                            getattr(block, "id", None),
                            getattr(block, "name", "")[len(BRIDGE_PREFIX) :],
                            json.dumps(getattr(block, "input", None) or {}, ensure_ascii=False),
                        )
                        for block in sendable
                    ]
                    logger.info(
                        "claude-sub: tool_calls ready ids=%s",
                        [call_id for call_id, _name, _args in self.tool_calls],
                    )
                    self.chunk_queue.put(_tool_calls_delta_chunk(self.model, self.tool_calls))
                    self.chunk_queue.put(_usage_chunk(self.model, self.last_call_usage))
                    raise PauseTurn()
        elif kind == "ResultMessage":
            self.result_message = message


class _ChatCompletions:
    def __init__(self, client: "ClaudeSubClient") -> None:
        self._client = client

    def create(self, **kwargs: Any) -> Any:
        return self._client._create_chat_completion(**kwargs)


class _ChatNamespace:
    def __init__(self, client: "ClaudeSubClient") -> None:
        self.completions = _ChatCompletions(client)


class ClaudeSubClient:
    """Minimal OpenAI-client-compatible facade for the claude-sub provider."""

    # This shim drives an SDK subprocess directly, so it is already a
    # complete client (never re-dispatch it through a wire adapter) and is
    # safe to use from async code as-is.
    HERMES_SKIP_TRANSPORT_WRAP = True
    HERMES_SKIP_ASYNC_WRAP = True

    def __init__(
        self,
        *,
        api_key: str | None = None,
        base_url: str | None = None,
        timeout: float | None = None,
        default_headers: dict[str, str] | None = None,
        **_ignored: Any,
    ) -> None:
        self.api_key = api_key or "claude-sub"
        self.base_url = base_url or MARKER_BASE_URL
        self._default_headers = dict(default_headers or {})
        self.is_closed = False
        self.chat = _ChatNamespace(self)
        self._settings = load_settings()
        self._turns: dict[str, _Turn] = {}
        self._turns_lock = threading.Lock()

    def close(self) -> None:
        self.is_closed = True
        with self._turns_lock:
            turns = list(self._turns.values())
            self._turns.clear()
        for turn in turns:
            self._abort_turn(turn)

    # ---------- turn bookkeeping ----------

    def _abort_turn(self, turn: _Turn) -> None:
        if turn.orphan_timer is not None:
            turn.orphan_timer.cancel()
            turn.orphan_timer = None
        self._cancel_idle_timer(turn)
        turn.session.request_interrupt()
        with turn.lock:
            pending = list(turn.pending.values())
            unbound = [fut for _name, _key, fut in turn.unbound]
            turn.pending.clear()
            turn.outstanding.clear()
            turn.hook_seen.clear()
            turn.unbound.clear()
            turn.results.clear()
            turn.handled_ids.clear()
            turn.cli_resolved.clear()
        for fut in pending:
            fut.cancel()
        for fut in unbound:
            fut.cancel()
        turn.session.close()

    def _discard_turn(self, session_key: str, turn: _Turn) -> None:
        with self._turns_lock:
            if self._turns.get(session_key) is turn:
                self._turns.pop(session_key, None)
        self._abort_turn(turn)

    def _arm_orphan_timer(self, session_key: str, turn: _Turn) -> None:
        def _on_orphan() -> None:
            logger.warning(
                "claude-sub: open turn (session_key=%s) got no continuation within %.0fs "
                "— interrupting",
                session_key,
                self._settings.orphan_timeout,
            )
            self._discard_turn(session_key, turn)

        timer = threading.Timer(self._settings.orphan_timeout, _on_orphan)
        timer.daemon = True
        turn.orphan_timer = timer
        timer.start()

    def _cancel_idle_timer(self, turn: _Turn) -> None:
        if turn.idle_timer is not None:
            turn.idle_timer.cancel()
            turn.idle_timer = None

    def _arm_idle_timer(self, session_key: str, turn: _Turn) -> None:
        def _on_idle_expired() -> None:
            with self._turns_lock:
                if self._turns.get(session_key) is not turn or turn.state != "idle":
                    return
                self._turns.pop(session_key, None)
            logger.info(
                "claude-sub: session_key=%s idle session TTL expired; closing", session_key
            )
            turn.session.close()

        timer = threading.Timer(self._settings.idle_session_ttl, _on_idle_expired)
        timer.daemon = True
        turn.idle_timer = timer
        timer.start()

    def _on_turn_stopped(
        self, turn: _Turn, session_key: str, *, messages: list[dict], final_text: str
    ) -> None:
        """Handle a clean ``stop``: keep the session warm (idle) or close it.

        A session is kept warm only when Hermes core gave it an explicit
        ``hermes_session_id`` (so we know a follow-up will address it by the
        same key — see D35-2) and warm retention is enabled
        (``idle_session_ttl > 0``). Otherwise this preserves the v0.1-D
        behavior of closing the session immediately.
        """
        settings = self._settings
        if turn.has_session_id and settings.idle_session_ttl > 0:
            turn.seen_count = len(messages) + 1
            turn.last_reply_text = (final_text or "").strip()
            turn.state = "idle"
            self._arm_idle_timer(session_key, turn)
            logger.info(
                "claude-sub: session_key=%s turn idle (session kept warm), ttl=%.0fs",
                session_key,
                settings.idle_session_ttl,
            )
            return
        with self._turns_lock:
            if self._turns.get(session_key) is turn:
                self._turns.pop(session_key, None)
        turn.session.close()

    def _resolve_pending(self, turn: _Turn, tail: list[dict]) -> None:
        for message in tail:
            call_id = message.get("tool_call_id")
            text = convert.text_from_content(message.get("content"))
            payload = {"content": [{"type": "text", "text": text}], "is_error": False}
            with turn.lock:
                fut = turn.pending.pop(call_id, None)
            if fut is not None:
                if fut.done():
                    continue
                logger.info("claude-sub: continuation resolving pending tool call %s", call_id)
                fut.set_result(payload)
            else:
                with turn.lock:
                    turn.results[call_id] = payload
                logger.info(
                    "claude-sub: continuation result for %s stashed; handler not invoked yet",
                    call_id,
                )
        with turn.lock:
            turn.outstanding.clear()

    def _build_session(
        self,
        *,
        model: str | None,
        reasoning_effort: str | None,
        mcp_servers: dict | None,
        allowed_tools: list | None,
        hooks: dict | None,
        max_turns: int | None,
    ) -> SdkSession:
        settings = self._settings
        cwd = os.getcwd()

        def _options_factory() -> Any:
            return build_options(
                model=model,
                reasoning_effort=reasoning_effort,
                identity_append=settings.identity_append,
                cwd=cwd,
                mcp_servers=mcp_servers,
                allowed_tools=allowed_tools,
                hooks=hooks,
                max_turns=max_turns,
            )

        def _transport_factory(options: Any) -> Any:
            from .env import build_sanitized_transport

            return build_sanitized_transport(options)

        return SdkSession(
            options_factory=_options_factory,
            transport_factory=_transport_factory,
            start_timeout=settings.start_timeout,
        )

    def _open_new_turn(
        self,
        session_key: str,
        *,
        tools: list[dict] | None,
        model: str | None,
        reasoning_effort: str | None,
        has_session_id: bool,
        system_text: str,
    ) -> _Turn:
        settings = self._settings
        turn = _Turn(
            session=None,
            start_timeout=settings.start_timeout,
            state="open",
            model=model,
            reasoning_effort=reasoning_effort,
            tools_key=_tools_key(tools),
            system_hash=_system_hash(system_text),
            has_session_id=has_session_id,
        )
        mcp_servers = None
        allowed_tools = None
        hooks = None
        if tools:
            server, allowed_tools = bridge.build_bridge(tools, _make_on_call(turn))
            mcp_servers = {"hermes": server}
            hooks = bridge.build_hooks(
                on_pre_tool_use=_make_on_pre_tool_use(turn),
                on_post_tool_use=_make_on_post_tool_use(turn),
            )
        turn.session = self._build_session(
            model=model,
            reasoning_effort=reasoning_effort,
            mcp_servers=mcp_servers,
            allowed_tools=allowed_tools,
            hooks=hooks,
            max_turns=None if tools else 1,
        )
        with self._turns_lock:
            self._turns[session_key] = turn
        return turn

    # ---------- non-streaming completion building ----------

    def _build_stop_completion(self, projector, *, model: str | None) -> Any:
        text = "".join(projector.text_parts)
        reasoning_text = "\n".join(projector.thinking_parts) if projector.thinking_parts else None
        usage = _usage_namespace(convert.usage_from_assistant(projector.last_call_usage))
        assistant_message = SimpleNamespace(
            content=text,
            tool_calls=None,
            reasoning=reasoning_text,
            reasoning_content=reasoning_text,
            reasoning_details=None,
        )
        choice = SimpleNamespace(message=assistant_message, finish_reason="stop")
        return SimpleNamespace(choices=[choice], usage=usage, model=model or "claude-sub")

    def _build_tool_calls_completion(self, projector, *, model: str | None) -> Any:
        from agent.acp_openai_bridge import build_openai_tool_call

        text = "".join(projector.text_parts) or None
        reasoning_text = "\n".join(projector.thinking_parts) if projector.thinking_parts else None
        usage = _usage_namespace(convert.usage_from_assistant(projector.last_call_usage))
        tool_calls = [
            build_openai_tool_call(call_id=call_id, name=name, arguments=arguments)
            for call_id, name, arguments in projector.tool_calls
        ]
        assistant_message = SimpleNamespace(
            content=text,
            tool_calls=tool_calls,
            reasoning=reasoning_text,
            reasoning_content=reasoning_text,
            reasoning_details=None,
        )
        choice = SimpleNamespace(message=assistant_message, finish_reason="tool_calls")
        return SimpleNamespace(choices=[choice], usage=usage, model=model or "claude-sub")

    def _finish(
        self,
        turn: _Turn,
        projector,
        session_key: str,
        *,
        model: str | None,
        messages: list[dict],
    ) -> Any:
        if projector.tool_calls is not None:
            logger.info(
                "claude-sub: session_key=%s returning tool_calls response (%d call(s)); turn left open",
                session_key,
                len(projector.tool_calls),
            )
            turn.state = "paused"
            self._arm_orphan_timer(session_key, turn)
            return self._build_tool_calls_completion(projector, model=model)
        if projector.result_message is not None:
            status = errors.classify_result(projector.result_message)
            if status is not None:
                self._discard_turn(session_key, turn)
                result_text = getattr(projector.result_message, "result", "") or ""
                reason = "rate-limit" if status == 429 else "error"
                errors.raise_status(status, errors.error_message_for(reason, result_text))
        logger.info("claude-sub: session_key=%s turn finished (finish_reason=stop)", session_key)
        final_text = "".join(projector.text_parts)
        self._on_turn_stopped(turn, session_key, messages=messages, final_text=final_text)
        return self._build_stop_completion(projector, model=model)

    # ---------- streaming ----------

    def _create_stream(
        self,
        session_key: str,
        turn: _Turn,
        *,
        is_continuation: bool,
        prompt: str | None,
        model: str,
        settings: Any,
        expect_tools: bool,
        messages: list[dict],
    ):
        chunk_queue: "queue.Queue" = queue.Queue()
        projector = _StreamingProjector(
            turn=turn, expect_tools=expect_tools, chunk_queue=chunk_queue, model=model
        )
        result_holder: dict[str, Any] = {}

        def _worker() -> None:
            try:
                if is_continuation:
                    turn.session.continue_turn(
                        on_message=projector,
                        timeout=settings.turn_timeout,
                        stall_timeout=settings.stall_timeout,
                        stall_exempt=lambda: bool(turn.pending),
                    )
                else:
                    turn.session.run_turn(
                        prompt,
                        on_message=projector,
                        timeout=settings.turn_timeout,
                        stall_timeout=settings.stall_timeout,
                        stall_exempt=lambda: bool(turn.pending),
                    )
            except TimeoutError as exc:
                result_holder["error"] = exc
                chunk_queue.put(_STREAM_DONE)
                return
            except Exception as exc:  # noqa: BLE001 - mapped in the generator
                result_holder["error"] = exc
                chunk_queue.put(_STREAM_DONE)
                return

            if projector.tool_calls is not None:
                pass  # tool_calls + usage chunks already queued by the projector
            elif projector.result_message is not None:
                status = errors.classify_result(projector.result_message)
                if status is not None:
                    result_text = getattr(projector.result_message, "result", "") or ""
                    reason = "rate-limit" if status == 429 else "error"
                    result_holder["mapped_status"] = status
                    result_holder["mapped_message"] = errors.error_message_for(
                        reason, result_text
                    )
                else:
                    chunk_queue.put(_stop_chunk(model))
                    chunk_queue.put(_usage_chunk(model, projector.last_call_usage))
            else:
                chunk_queue.put(_stop_chunk(model))
                chunk_queue.put(_usage_chunk(model, projector.last_call_usage))
            chunk_queue.put(_STREAM_DONE)

        thread = threading.Thread(target=_worker, name="claude-sub-stream", daemon=True)
        thread.start()

        def _generator():
            try:
                while True:
                    item = chunk_queue.get()
                    if item is _STREAM_DONE:
                        break
                    yield item
            finally:
                thread.join(timeout=15.0)

            if "error" in result_holder:
                exc = result_holder["error"]
                self._discard_turn(session_key, turn)
                if isinstance(exc, TimeoutError):
                    errors.raise_status(504, errors.error_message_for("timeout", str(exc)))
                errors.raise_status(503, errors.error_message_for("sdk-error", str(exc)))
            if "mapped_status" in result_holder:
                self._discard_turn(session_key, turn)
                errors.raise_status(result_holder["mapped_status"], result_holder["mapped_message"])
            if projector.tool_calls is not None:
                turn.state = "paused"
                self._arm_orphan_timer(session_key, turn)
            else:
                final_text = "".join(projector.text_parts)
                self._on_turn_stopped(turn, session_key, messages=messages, final_text=final_text)

        return _generator()

    # ---------- entry point ----------

    def _resolve_session_key(self, messages: list[dict], extra_body: dict | None) -> str:
        if isinstance(extra_body, dict):
            session_key = extra_body.get("hermes_session_id")
            if session_key:
                return session_key
        system_text, _last_user_text, _prior = convert.split_messages(messages)
        first_user = convert.first_user_text(messages)
        return _derive_session_key(system_text, first_user)

    def _create_chat_completion(
        self,
        *,
        model: str | None = None,
        messages: list[dict[str, Any]] | None = None,
        tools: list[dict[str, Any]] | None = None,
        tool_choice: Any = None,
        stream: bool = False,
        extra_body: dict[str, Any] | None = None,
        reasoning_effort: str | None = None,
        timeout: float | None = None,
        **_ignored: Any,
    ) -> Any:
        messages = messages or []
        settings = self._settings
        session_key = self._resolve_session_key(messages, extra_body)
        has_session_id = isinstance(extra_body, dict) and bool(extra_body.get("hermes_session_id"))

        # The idle-timer callback (``_arm_idle_timer``'s ``_on_idle_expired``)
        # pops the turn from ``self._turns`` and closes its session under
        # ``_turns_lock`` the instant it sees ``state == "idle"``. The lookup,
        # the warm-followup decision, and the "claim it" transition
        # (state -> "open") must all happen inside that same lock, or the
        # timer can fire in the gap between deciding "warm" and recording the
        # claim, close the session out from under this call, and leave it
        # calling ``run_turn``/``continue_turn`` on an already-closed session
        # (D35 review, run 180). ``_is_warm_followup`` is pure (no I/O), so
        # holding the lock across it is safe. Re-fetching by key inside the
        # lock also means a timer that already won the race (already popped
        # the turn) is simply seen as a miss here and falls through to the
        # cold-start path below.
        with self._turns_lock:
            turn = self._turns.get(session_key)
            is_warm_followup = False
            if turn is not None and turn.state == "idle":
                if _is_warm_followup(
                    messages,
                    turn,
                    model=model,
                    reasoning_effort=reasoning_effort,
                    tools=tools,
                    extra_body=extra_body,
                ):
                    is_warm_followup = True
                    turn.state = "open"

        tail = None
        if not is_warm_followup and turn is not None and turn.state == "paused":
            tail = _classify_continuation(messages, turn)
        is_continuation = tail is not None

        prompt: str | None = None
        if is_continuation:
            self._resolve_pending(turn, tail)
            if turn.orphan_timer is not None:
                turn.orphan_timer.cancel()
                turn.orphan_timer = None
            turn.state = "open"
        elif is_warm_followup:
            self._cancel_idle_timer(turn)
            tail_messages = messages[turn.seen_count :]
            prompt = "\n\n".join(
                convert.text_from_content(message.get("content")) for message in tail_messages
            )
        else:
            if turn is not None:
                self._discard_turn(session_key, turn)
            prompt = convert.build_prompt(messages, bootstrap_max_chars=settings.bootstrap_max_chars)
            system_text, _last_user_text, _prior = convert.split_messages(messages)
            turn = self._open_new_turn(
                session_key,
                tools=tools,
                model=model,
                reasoning_effort=reasoning_effort,
                has_session_id=has_session_id,
                system_text=system_text,
            )

        logger.info(
            "claude-sub: create() session_key=%s continuation=%s warm=%s state=%s tools=%s stream=%s",
            session_key,
            is_continuation,
            is_warm_followup,
            turn.state,
            bool(tools),
            stream,
        )

        if stream:
            return self._create_stream(
                session_key,
                turn,
                is_continuation=is_continuation,
                prompt=prompt,
                model=model or "claude-sub",
                settings=settings,
                expect_tools=bool(tools),
                messages=messages,
            )

        projector = _InversionProjector(turn=turn, expect_tools=bool(tools))
        try:
            if is_continuation:
                turn.session.continue_turn(
                    on_message=projector,
                    timeout=settings.turn_timeout,
                    stall_timeout=settings.stall_timeout,
                    stall_exempt=lambda: bool(turn.pending),
                )
            else:
                turn.session.run_turn(
                    prompt,
                    on_message=projector,
                    timeout=settings.turn_timeout,
                    stall_timeout=settings.stall_timeout,
                    stall_exempt=lambda: bool(turn.pending),
                )
        except TimeoutError as exc:
            self._discard_turn(session_key, turn)
            errors.raise_status(504, errors.error_message_for("timeout", str(exc)))
        except Exception as exc:  # noqa: BLE001 - mapped to a wire-shaped error below
            self._discard_turn(session_key, turn)
            errors.raise_status(503, errors.error_message_for("sdk-error", str(exc)))

        return self._finish(turn, projector, session_key, model=model, messages=messages)


def _make_on_pre_tool_use(turn: _Turn):
    def on_pre_tool_use(tool_use_id: str, short_name: str, tool_input: dict) -> None:
        key = args_key(tool_input)
        with turn.lock:
            turn.hook_seen.append((tool_use_id, short_name, key))

    return on_pre_tool_use


def _make_on_post_tool_use(turn: _Turn):
    def on_post_tool_use(tool_use_id: str, short_name: str, tool_response: Any, failed: bool) -> None:
        summary = repr(tool_response)
        if len(summary) > 200:
            summary = summary[:200] + "…"
        with turn.lock:
            handled = tool_use_id in turn.handled_ids
            if not handled:
                turn.cli_resolved.add(tool_use_id)
            # PostToolUse settles this id; a stale hook_seen entry for it must
            # not win a later on_call() match (that would bind a live call to
            # a dead id and block forever — see v0.1-D review, run 176).
            turn.hook_seen[:] = [entry for entry in turn.hook_seen if entry[0] != tool_use_id]
        logger.info(
            "claude-sub: PostToolUse id=%s name=%s handled=%s failed=%s response=%s",
            tool_use_id,
            short_name,
            handled,
            failed,
            summary,
        )

    return on_post_tool_use


__all__ = ["ClaudeSubClient", "MARKER_BASE_URL"]
