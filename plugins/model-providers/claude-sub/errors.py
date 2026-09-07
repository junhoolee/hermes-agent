"""Error classification and mapping for the claude-sub provider.

Maps claude-agent-sdk failures onto the ``openai`` exception hierarchy so
Hermes' existing error classifier (``agent/error_classifier.py``) recognizes
them as rate-limit / server / timeout failures without any claude-sub-specific
branch in core.
"""

from __future__ import annotations

from typing import Any

import httpx
import openai

_FAKE_URL = "claude-sub://sdk"

_RATE_LIMIT_MARKERS = (
    "rate limit",
    "rate_limit",
    "usage limit",
    "quota",
    "overloaded",
    "529",
    "too many requests",
)


def raise_status(status: int, message: str) -> None:
    """Raise the ``openai`` exception matching *status* for *message*."""
    request = httpx.Request("POST", _FAKE_URL)
    response = httpx.Response(status, request=request)
    if status == 429:
        raise openai.RateLimitError(message, response=response, body=None)
    raise openai.APIStatusError(message, response=response, body=None)


def classify_result(result_message: Any) -> int | None:
    """Return the HTTP-like status a ``ResultMessage`` implies, or ``None``.

    ``None`` means the turn did not report an error. 429 for anything that
    looks like a rate/usage limit, 503 for any other reported error.
    """
    is_error = bool(getattr(result_message, "is_error", False))
    api_error_status = getattr(result_message, "api_error_status", None)
    if api_error_status == 429:
        return 429
    if not is_error:
        return None

    parts: list[str] = []
    result_text = getattr(result_message, "result", None)
    if isinstance(result_text, str):
        parts.append(result_text)
    errors = getattr(result_message, "errors", None)
    if isinstance(errors, list):
        parts.extend(str(e) for e in errors)
    combined = " ".join(parts).lower()
    if any(marker in combined for marker in _RATE_LIMIT_MARKERS):
        return 429
    return 503


def error_message_for(reason: str, sdk_text: str = "") -> str:
    """Build the ``claude-sub: <reason>: <text>`` message, capped at 500 chars."""
    text = (sdk_text or "").strip()[:500]
    if text:
        return f"claude-sub: {reason}: {text}"
    return f"claude-sub: {reason}"


__all__ = ["raise_status", "classify_result", "error_message_for"]
