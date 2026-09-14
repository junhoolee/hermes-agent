"""Synchronous session runner for the claude-sub provider.

``ClaudeSDKClient`` is asyncio-only; the plugin's ``ClaudeSubClient`` is
synchronous. This module owns the bridge: one dedicated event-loop thread per
``SdkSession``, driving a single ``ClaudeSDKClient``. The prompt passed to
``run_turn``/``continue_turn`` is opaque here — a plain string or a
stream-json ``AsyncIterable[dict]`` (``convert.StreamPrompt``, used when the
turn carries an image) — and is forwarded to ``client.query()`` as-is.

v0.1-B adds pause/continue: an ``on_message`` callback can raise
``PauseTurn`` to stop draining early (a tool-call boundary) while leaving the
SDK turn open — the background pump keeps running and buffering further
messages, and a later ``continue_turn()`` call resumes draining the same
turn without issuing a new ``query()``.

Adapted from ``agent/transports/claude_agent_session.py`` with the session
store / resume / materialization machinery removed: this plugin never
resumes a prior transcript across process restarts.
"""

from __future__ import annotations

import asyncio
import logging
import os
import queue
import threading
import time
from typing import Any, Callable, Optional

logger = logging.getLogger(__name__)

DEFAULT_IDENTITY_APPEND = (
    "You are running inside Hermes Agent. The operating instructions in the "
    "first user message define your identity and behavior; follow them over "
    "this harness preset, and do not describe yourself as Claude Code."
)

_VALID_EFFORTS = {"low", "medium", "high", "xhigh", "max"}


class SdkSessionError(RuntimeError):
    """Raised for session-level failures (startup, teardown, wedged turn)."""


class PauseTurn(Exception):
    """Raised by an ``on_message`` callback to stop draining early.

    The SDK turn stays open — the background pump keeps buffering further
    messages into the same inbox. The caller retrieves them again via
    ``continue_turn()`` rather than starting a fresh ``query()``.
    """


def is_result_message(message: Any) -> bool:
    """True when *message* is the SDK's terminal ``ResultMessage``.

    Matched by class name so this behaves identically against the real
    optional extra and the stand-in the test suite installs when it is
    absent.
    """
    return type(message).__name__ == "ResultMessage"


def build_options(
    *,
    model: Optional[str] = None,
    reasoning_effort: Optional[str] = None,
    identity_append: str = "",
    cwd: Optional[str] = None,
    stderr: Optional[Callable[[str], None]] = None,
    mcp_servers: Optional[dict] = None,
    allowed_tools: Optional[list] = None,
    hooks: Optional[dict] = None,
    max_turns: Optional[int] = 1,
) -> Any:
    """Build ``ClaudeAgentOptions`` for this plugin.

    Two shapes: a tool-less single turn (``mcp_servers`` omitted — the
    default, used for aux one-shot calls) and a bridged turn (``mcp_servers``
    given — ``tools`` is deliberately left unspecified so the CLI's built-in
    tool context stays intact for its billing classifier; a PreToolUse hook
    is what actually keeps them from running, see ``bridge.py``).
    """
    from claude_agent_sdk import ClaudeAgentOptions

    kwargs: dict = dict(
        system_prompt={
            "type": "preset",
            "preset": "claude_code",
            "append": identity_append or DEFAULT_IDENTITY_APPEND,
        },
        setting_sources=[],
        strict_mcp_config=True,
        cwd=cwd or os.getcwd(),
        env={},
        include_partial_messages=True,
        stderr=stderr or (lambda line: logger.debug("claude-sub stderr: %s", line)),
    )
    if mcp_servers:
        kwargs["mcp_servers"] = mcp_servers
        kwargs["allowed_tools"] = list(allowed_tools or [])
        if hooks:
            kwargs["hooks"] = hooks
    else:
        kwargs["mcp_servers"] = {}
        kwargs["allowed_tools"] = []
        kwargs["tools"] = []
    if max_turns is not None:
        kwargs["max_turns"] = max_turns
    if model:
        kwargs["model"] = model
    if reasoning_effort in _VALID_EFFORTS:
        kwargs["effort"] = reasoning_effort
    return ClaudeAgentOptions(**kwargs)


class SdkSession:
    """One ``ClaudeSDKClient`` on one owned event-loop thread."""

    def __init__(
        self,
        *,
        options_factory: Callable[[], Any],
        client_factory: Optional[Callable[..., Any]] = None,
        transport_factory: Optional[Callable[[Any], Any]] = None,
        start_timeout: float = 60.0,
        close_timeout: float = 15.0,
    ) -> None:
        self._options_factory = options_factory
        self._client_factory = client_factory
        self._transport_factory = transport_factory
        self._start_timeout = start_timeout
        self._close_timeout = close_timeout

        self._lock = threading.RLock()
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._thread: Optional[threading.Thread] = None
        self._client: Any = None
        self._closed = False

        # An open, paused turn: the background pump keeps running and
        # buffering into this inbox until continue_turn() drains it further.
        self._pending_inbox: Optional["queue.Queue[tuple[str, Any]]"] = None
        self._pending_future: Any = None

    @property
    def started(self) -> bool:
        return self._client is not None

    @property
    def closed(self) -> bool:
        return self._closed

    @property
    def has_open_turn(self) -> bool:
        return self._pending_inbox is not None

    def ensure_started(self) -> None:
        """Spawn the loop thread and connect the client. Idempotent."""
        with self._lock:
            if self._closed:
                raise SdkSessionError("claude-sub session is closed; build a new one.")
            if self._client is not None:
                return

            loop = asyncio.new_event_loop()
            ready = threading.Event()
            thread = threading.Thread(
                target=self._run_loop,
                args=(loop, ready),
                name="claude-sub-sdk",
                daemon=True,
            )
            thread.start()
            if not ready.wait(self._start_timeout):
                loop.call_soon_threadsafe(loop.stop)
                raise SdkSessionError(
                    f"claude-sub event loop failed to start within {self._start_timeout:.0f}s."
                )
            self._loop = loop
            self._thread = thread

            try:
                self._client = self._submit(self._connect(), timeout=self._start_timeout)
            except BaseException:
                self._teardown_loop()
                raise

    def close(self) -> None:
        """Disconnect the client, stop the loop, join the thread. Idempotent."""
        with self._lock:
            if self._closed:
                return
            self._closed = True
            client = self._client
            self._client = None
            pending_future = self._pending_future
            self._pending_inbox = None
            self._pending_future = None

        if pending_future is not None:
            pending_future.cancel()
        if client is not None:
            try:
                self._submit(client.disconnect(), timeout=self._close_timeout)
            except Exception:
                logger.debug("claude-sub disconnect failed", exc_info=True)
        self._teardown_loop()

    def request_interrupt(self) -> bool:
        """Ask the CLI to abort the active turn. Never raises."""
        client = self._client
        if client is None or self._closed:
            return False
        try:
            self._submit(client.interrupt(), timeout=30.0)
            return True
        except Exception:
            logger.debug("claude-sub interrupt failed", exc_info=True)
            return False

    def run_turn(
        self,
        prompt: Any,
        *,
        on_message: Callable[[Any], None],
        timeout: Optional[float] = None,
        stall_timeout: Optional[float] = None,
        stall_exempt: Optional[Callable[[], bool]] = None,
    ) -> int:
        """Send *prompt* (a fresh ``query()``) and drain until quiet or paused.

        *prompt* is a str or a stream-json frame's ``AsyncIterable[dict]``
        (``convert.StreamPrompt``) — passed to ``client.query()`` unchanged.

        Raises ``TimeoutError`` on a blown deadline or a stall, and re-raises
        whatever the SDK raised. If *on_message* raises ``PauseTurn``, this
        returns normally with the SDK turn left open — see ``continue_turn``.
        """
        if self._pending_inbox is not None:
            raise SdkSessionError(
                "claude-sub session already has an open turn; call continue_turn()."
            )
        self.ensure_started()
        inbox: "queue.Queue[tuple[str, Any]]" = queue.Queue()
        future = asyncio.run_coroutine_threadsafe(
            self._pump_turn(prompt, inbox), self._require_loop()
        )
        return self._drain(
            inbox,
            future,
            on_message=on_message,
            timeout=timeout,
            stall_timeout=stall_timeout,
            stall_exempt=stall_exempt,
        )

    def continue_turn(
        self,
        *,
        on_message: Callable[[Any], None],
        timeout: Optional[float] = None,
        stall_timeout: Optional[float] = None,
        stall_exempt: Optional[Callable[[], bool]] = None,
    ) -> int:
        """Resume draining a turn previously paused by ``PauseTurn``. No new ``query()``."""
        inbox = self._pending_inbox
        future = self._pending_future
        if inbox is None or future is None:
            raise SdkSessionError("claude-sub session has no open turn to continue.")
        return self._drain(
            inbox,
            future,
            on_message=on_message,
            timeout=timeout,
            stall_timeout=stall_timeout,
            stall_exempt=stall_exempt,
        )

    # ---------- internals ----------

    def _drain(
        self,
        inbox: "queue.Queue[tuple[str, Any]]",
        future: Any,
        *,
        on_message: Callable[[Any], None],
        timeout: Optional[float],
        stall_timeout: Optional[float],
        stall_exempt: Optional[Callable[[], bool]],
    ) -> int:
        turn_timeout = timeout if timeout is not None else 1800.0
        deadline = time.monotonic() + turn_timeout
        last_activity = time.monotonic()
        delivered = 0
        paused = False
        try:
            while True:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    self.request_interrupt()
                    raise TimeoutError(
                        f"claude-sub turn exceeded {turn_timeout:.0f}s without completing."
                    )
                try:
                    kind, payload = inbox.get(timeout=min(remaining, 0.25))
                except queue.Empty:
                    if stall_timeout is not None:
                        now = time.monotonic()
                        if stall_exempt is not None and stall_exempt():
                            last_activity = now
                        elif now - last_activity > stall_timeout:
                            self.request_interrupt()
                            raise TimeoutError(
                                f"claude-sub turn received no SDK messages for "
                                f"{stall_timeout:.0f}s (stalled)."
                            )
                    continue
                last_activity = time.monotonic()
                if kind == "message":
                    delivered += 1
                    try:
                        on_message(payload)
                    except PauseTurn:
                        paused = True
                        break
                    continue
                if kind == "error":
                    raise payload
                break
        except BaseException:
            if not paused:
                future.cancel()
                self._pending_inbox = None
                self._pending_future = None
            raise
        if paused:
            self._pending_inbox = inbox
            self._pending_future = future
        else:
            future.result(timeout=self._close_timeout)
            self._pending_inbox = None
            self._pending_future = None
        return delivered

    def _run_loop(self, loop: asyncio.AbstractEventLoop, ready: threading.Event) -> None:
        asyncio.set_event_loop(loop)
        try:
            loop.call_soon(ready.set)
            loop.run_forever()
        finally:
            try:
                loop.close()
            except Exception:
                pass

    def _require_loop(self) -> asyncio.AbstractEventLoop:
        loop = self._loop
        if loop is None or loop.is_closed():
            raise SdkSessionError("claude-sub session has no running event loop.")
        return loop

    def _submit(self, coro: Any, *, timeout: float) -> Any:
        future = asyncio.run_coroutine_threadsafe(coro, self._require_loop())
        try:
            return future.result(timeout=timeout)
        except TimeoutError:
            future.cancel()
            raise

    async def _connect(self) -> Any:
        factory = self._client_factory
        if factory is None:
            from claude_agent_sdk import ClaudeSDKClient

            factory = ClaudeSDKClient
        options = self._options_factory()
        kwargs: dict = {"options": options}
        if self._transport_factory is not None:
            transport = self._transport_factory(options)
            if transport is not None:
                kwargs["transport"] = transport
        client = factory(**kwargs)
        await client.connect()
        return client

    async def _pump_turn(self, prompt: Any, inbox: "queue.Queue[tuple[str, Any]]") -> None:
        client = self._client
        if client is None:
            inbox.put(("error", SdkSessionError("claude-sub session is closed.")))
            return
        iterator = None
        try:
            await client.query(prompt)
            iterator = client.receive_messages().__aiter__()
            saw_result = False
            while True:
                try:
                    if saw_result:
                        message = await asyncio.wait_for(iterator.__anext__(), 1.0)
                    else:
                        message = await iterator.__anext__()
                except (StopAsyncIteration, asyncio.TimeoutError):
                    break
                if is_result_message(message):
                    saw_result = True
                inbox.put(("message", message))
            inbox.put(("done", None))
        except BaseException as exc:  # noqa: BLE001 - relayed to the caller
            inbox.put(("error", exc))
        finally:
            if iterator is not None:
                aclose = getattr(iterator, "aclose", None)
                if aclose is not None:
                    try:
                        await aclose()
                    except Exception:
                        logger.debug("claude-sub stream aclose failed", exc_info=True)

    def _teardown_loop(self) -> None:
        with self._lock:
            loop = self._loop
            thread = self._thread
            self._loop = None
            self._thread = None
        if loop is not None and not loop.is_closed():
            try:
                loop.call_soon_threadsafe(loop.stop)
            except RuntimeError:
                pass
        if thread is not None and thread is not threading.current_thread():
            thread.join(timeout=self._close_timeout)
            if thread.is_alive():
                logger.warning(
                    "claude-sub loop thread did not exit within %.0fs", self._close_timeout
                )


__all__ = [
    "SdkSession",
    "SdkSessionError",
    "PauseTurn",
    "DEFAULT_IDENTITY_APPEND",
    "build_options",
    "is_result_message",
]
