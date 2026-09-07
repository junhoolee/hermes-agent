"""OpenAI-client-shaped facade over the claude-agent-sdk for claude-sub.

v0.1-A is a one-shot, tool-less path: every ``create()`` call spins up a
fresh, single-turn ``SdkSession`` (no tools, ``max_turns=1``), waits for the
text response, and tears the session down. Session reuse, continuation, and a
real streaming path are card B's job — see the plugin README.
"""

from __future__ import annotations

import hashlib
import logging
import os
from types import SimpleNamespace
from typing import Any

from . import convert, errors
from .config import load_settings
from .session import SdkSession, build_options

logger = logging.getLogger(__name__)

MARKER_BASE_URL = "claude-sub://sdk"


class _TurnProjector:
    """Accumulates one turn's text/thinking/usage/result from SDK messages."""

    def __init__(self) -> None:
        self.text_parts: list[str] = []
        self.thinking_parts: list[str] = []
        self.last_call_usage: dict | None = None
        self.result_message: Any = None

    def __call__(self, message: Any) -> None:
        kind = type(message).__name__
        if kind == "AssistantMessage":
            usage = getattr(message, "usage", None)
            if isinstance(usage, dict) and usage:
                self.last_call_usage = usage
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
        elif kind == "ResultMessage":
            self.result_message = message


def _derive_session_key(system_text: str, first_user_text: str) -> str:
    digest = hashlib.sha1(f"{system_text}\x00{first_user_text}".encode()).hexdigest()
    return digest[:16]


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

    def close(self) -> None:
        self.is_closed = True

    def _build_session(self, *, model: str | None, reasoning_effort: str | None) -> SdkSession:
        settings = self._settings
        cwd = os.getcwd()

        def _options_factory() -> Any:
            return build_options(
                model=model,
                reasoning_effort=reasoning_effort,
                identity_append=settings.identity_append,
                cwd=cwd,
            )

        def _transport_factory(options: Any) -> Any:
            from .env import build_sanitized_transport

            return build_sanitized_transport(options)

        return SdkSession(
            options_factory=_options_factory,
            transport_factory=_transport_factory,
            start_timeout=settings.start_timeout,
        )

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

        system_text, _last_user_text, _prior = convert.split_messages(messages)
        first_user = convert.first_user_text(messages)
        session_key = None
        if isinstance(extra_body, dict):
            session_key = extra_body.get("hermes_session_id")
        if not session_key:
            session_key = _derive_session_key(system_text, first_user)
        logger.debug("claude-sub turn (session_key=%s)", session_key)

        prompt = convert.build_prompt(messages, bootstrap_max_chars=settings.bootstrap_max_chars)
        projector = _TurnProjector()
        session = self._build_session(model=model, reasoning_effort=reasoning_effort)
        try:
            session.run_turn(
                prompt,
                on_message=projector,
                timeout=settings.turn_timeout,
                stall_timeout=settings.stall_timeout,
            )
        except TimeoutError as exc:
            errors.raise_status(504, errors.error_message_for("timeout", str(exc)))
        except Exception as exc:  # noqa: BLE001 - mapped to a wire-shaped error below
            errors.raise_status(503, errors.error_message_for("sdk-error", str(exc)))
        finally:
            session.close()

        if projector.result_message is not None:
            status = errors.classify_result(projector.result_message)
            if status is not None:
                result_text = getattr(projector.result_message, "result", "") or ""
                reason = "rate-limit" if status == 429 else "error"
                errors.raise_status(status, errors.error_message_for(reason, result_text))

        text = "".join(projector.text_parts)
        reasoning_text = "\n".join(projector.thinking_parts) if projector.thinking_parts else None
        usage_dict = convert.usage_from_assistant(projector.last_call_usage)

        usage = SimpleNamespace(
            prompt_tokens=usage_dict["prompt_tokens"],
            completion_tokens=usage_dict["completion_tokens"],
            total_tokens=usage_dict["total_tokens"],
            prompt_tokens_details=SimpleNamespace(cached_tokens=usage_dict["cached_tokens"]),
        )
        assistant_message = SimpleNamespace(
            content=text,
            tool_calls=None,
            reasoning=reasoning_text,
            reasoning_content=reasoning_text,
            reasoning_details=None,
        )
        choice = SimpleNamespace(message=assistant_message, finish_reason="stop")
        completion = SimpleNamespace(
            choices=[choice],
            usage=usage,
            model=model or "claude-sub",
        )
        if stream:
            from agent.acp_openai_bridge import completion_to_stream_chunks

            return completion_to_stream_chunks(completion)
        return completion


__all__ = ["ClaudeSubClient", "MARKER_BASE_URL"]
