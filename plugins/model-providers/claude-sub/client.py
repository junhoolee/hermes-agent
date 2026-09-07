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

See ``bridge.py`` for the in-process MCP server + Future plumbing, and
``session.py`` for the pause/continue primitive this relies on
(``PauseTurn`` / ``continue_turn``).
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

# Upper bound for reconciling a bridge handler call against its matching
# tool-use block when they arrive out of order (see ``_wait_for_pending``).
# Deliberately independent of ``settings.start_timeout`` (SDK session
# connect timeout) — this is a same-process handoff, not a subprocess
# startup, so it should resolve in milliseconds, not tens of seconds.
BRIDGE_BIND_TIMEOUT = 10.0


class BridgeBindTimeout(RuntimeError):
    """A bridge handler call and its tool-use block never reconciled in time."""


@dataclass
class _Turn:
    """One open Hermes<->SDK turn, keyed by session_key in ``ClaudeSubClient``."""

    session: Any
    start_timeout: float
    pending: dict = field(default_factory=dict)  # call_id -> concurrent.futures.Future
    expected_ids: list = field(default_factory=list)  # ordered [(call_id, name), ...]
    unbound: list = field(default_factory=list)  # ordered [(name, Future), ...]
    lock: threading.RLock = field(default_factory=threading.RLock)
    opened_at: float = field(default_factory=time.monotonic)
    orphan_timer: Any = None


def _derive_session_key(system_text: str, first_user_text: str) -> str:
    digest = hashlib.sha1(f"{system_text}\x00{first_user_text}".encode()).hexdigest()
    return digest[:16]


def _register_expected(turn: _Turn, tool_blocks: list) -> None:
    """Match each tool-use *block* against a bridge handler call, either way round.

    Handler and projector run on different threads and there is no
    guarantee which one reaches its half of the pair first (the SDK spawns
    the MCP tool call as soon as it appears on the control channel, often
    before the drain thread has projected the matching ``AssistantMessage``
    block — see the module docstring). If ``on_call`` already parked a
    Future for this block's short name in ``turn.unbound``, claim it
    directly into ``pending``; otherwise queue the expectation for
    ``on_call`` to find when it runs.
    """
    with turn.lock:
        for block in tool_blocks:
            name = getattr(block, "name", "") or ""
            short_name = name[len(BRIDGE_PREFIX):] if name.startswith(BRIDGE_PREFIX) else name
            call_id = getattr(block, "id", None)
            fut = None
            for index, (unbound_name, unbound_fut) in enumerate(turn.unbound):
                if unbound_name == short_name:
                    fut = unbound_fut
                    del turn.unbound[index]
                    break
            if fut is not None:
                turn.pending[call_id] = fut
            else:
                turn.expected_ids.append((call_id, short_name))


def _wait_for_pending(turn: _Turn, ids: list) -> None:
    """Block (briefly) until *ids* all have a Future registered in pending.

    ``on_call`` and ``_register_expected`` reconcile handler-first and
    projector-first arrival against each other under ``turn.lock`` (see
    their docstrings), so by the time either side reaches this call, *ids*
    should already be bound — this is a short poll for the remaining
    in-flight window between the two. Bounded by ``BRIDGE_BIND_TIMEOUT`` so
    a genuinely wedged handler can't hang the pump forever.
    """
    deadline = time.monotonic() + BRIDGE_BIND_TIMEOUT
    while True:
        with turn.lock:
            if all(i in turn.pending for i in ids):
                return
        if time.monotonic() >= deadline:
            logger.warning(
                "claude-sub: timed out waiting for bridge handler registration for %s", ids
            )
            raise BridgeBindTimeout(
                f"claude-sub: bridge handler registration timed out for {ids}"
            )
        time.sleep(0.01)


def _make_on_call(turn: _Turn):
    def on_call(name: str, _args: dict) -> "concurrent.futures.Future":
        fut: "concurrent.futures.Future" = concurrent.futures.Future()
        with turn.lock:
            call_id = None
            for index, (cid, expected_name) in enumerate(turn.expected_ids):
                if expected_name == name:
                    call_id = cid
                    del turn.expected_ids[index]
                    break
            if call_id is not None:
                turn.pending[call_id] = fut
            else:
                turn.unbound.append((name, fut))
                logger.debug(
                    "claude-sub: bridge handler for %s arrived before its tool_use block; parking",
                    name,
                )
        return fut

    return on_call


def _classify_continuation(messages: list[dict], turn: "_Turn | None") -> list[dict] | None:
    """Return the trailing tool-result messages if *messages* continues *turn*, else None."""
    if turn is None:
        return None
    with turn.lock:
        pending_ids = set(turn.pending.keys())
    if not pending_ids:
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

    if tail_ids != assistant_ids or tail_ids != pending_ids:
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
                _register_expected(self.turn, tool_blocks)
                ids = [getattr(b, "id", None) for b in tool_blocks]
                _wait_for_pending(self.turn, ids)
                self.tool_calls = [
                    (
                        getattr(block, "id", None),
                        getattr(block, "name", "")[len(BRIDGE_PREFIX):],
                        json.dumps(getattr(block, "input", None) or {}, ensure_ascii=False),
                    )
                    for block in tool_blocks
                ]
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
                            self.chunk_queue.put(_data_chunk(self.model, content=text))
                elif block_kind == "ToolUseBlock" and self.expect_tools:
                    name = getattr(block, "name", "") or ""
                    if name.startswith(BRIDGE_PREFIX):
                        tool_blocks.append(block)
            if tool_blocks:
                _register_expected(self.turn, tool_blocks)
                ids = [getattr(b, "id", None) for b in tool_blocks]
                _wait_for_pending(self.turn, ids)
                self.tool_calls = [
                    (
                        getattr(block, "id", None),
                        getattr(block, "name", "")[len(BRIDGE_PREFIX):],
                        json.dumps(getattr(block, "input", None) or {}, ensure_ascii=False),
                    )
                    for block in tool_blocks
                ]
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
        turn.session.request_interrupt()
        with turn.lock:
            pending = list(turn.pending.values())
            unbound = [fut for _name, fut in turn.unbound]
            turn.pending.clear()
            turn.expected_ids.clear()
            turn.unbound.clear()
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

    def _resolve_pending(self, turn: _Turn, tail: list[dict]) -> None:
        for message in tail:
            call_id = message.get("tool_call_id")
            with turn.lock:
                fut = turn.pending.pop(call_id, None)
            if fut is None or fut.done():
                continue
            text = convert.text_from_content(message.get("content"))
            logger.info("claude-sub: continuation resolving pending tool call %s", call_id)
            fut.set_result({"content": [{"type": "text", "text": text}], "is_error": False})

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
    ) -> _Turn:
        settings = self._settings
        turn = _Turn(session=None, start_timeout=settings.start_timeout)
        mcp_servers = None
        allowed_tools = None
        hooks = None
        if tools:
            server, allowed_tools = bridge.build_bridge(tools, _make_on_call(turn))
            mcp_servers = {"hermes": server}
            hooks = bridge.build_pretooluse_hooks()
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

    def _finish(self, turn: _Turn, projector, session_key: str, *, model: str | None) -> Any:
        if projector.tool_calls is not None:
            logger.info(
                "claude-sub: session_key=%s returning tool_calls response (%d call(s)); turn left open",
                session_key,
                len(projector.tool_calls),
            )
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
        with self._turns_lock:
            if self._turns.get(session_key) is turn:
                self._turns.pop(session_key, None)
        turn.session.close()
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
                self._arm_orphan_timer(session_key, turn)
            else:
                with self._turns_lock:
                    if self._turns.get(session_key) is turn:
                        self._turns.pop(session_key, None)
                turn.session.close()

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

        with self._turns_lock:
            turn = self._turns.get(session_key)

        tail = _classify_continuation(messages, turn)
        is_continuation = tail is not None

        prompt: str | None = None
        if is_continuation:
            self._resolve_pending(turn, tail)
            if turn.orphan_timer is not None:
                turn.orphan_timer.cancel()
                turn.orphan_timer = None
        else:
            if turn is not None:
                self._discard_turn(session_key, turn)
            prompt = convert.build_prompt(messages, bootstrap_max_chars=settings.bootstrap_max_chars)
            turn = self._open_new_turn(
                session_key, tools=tools, model=model, reasoning_effort=reasoning_effort
            )

        logger.info(
            "claude-sub: create() session_key=%s continuation=%s tools=%s stream=%s",
            session_key,
            is_continuation,
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

        return self._finish(turn, projector, session_key, model=model)


__all__ = ["ClaudeSubClient", "MARKER_BASE_URL"]
