"""Message <-> prompt conversion and usage accounting for claude-sub.

The claude-agent-sdk session speaks one text prompt per turn, not an OpenAI
``messages`` list. This module renders the Hermes conversation into that
prompt — including a coldstart's prior ``tool_calls``/``tool`` history, so
a model resuming mid-conversation can see what was already tried, with what
arguments, and what came back, instead of repeating the same call blind —
and turns the SDK's per-call usage dict into OpenAI-shaped token counts.
"""

from __future__ import annotations

from typing import Any

BOOTSTRAP_MAX_MESSAGES = 200
BOOTSTRAP_TOOL_ARGS_MAX_CHARS = 500
BOOTSTRAP_TOOL_RESULT_MAX_CHARS = 2000


def _text_from_content(content: Any) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for part in content:
            if not isinstance(part, dict):
                continue
            part_type = part.get("type")
            if part_type == "text":
                parts.append(str(part.get("text") or ""))
            elif part_type == "image_url":
                parts.append("[image omitted]")
        return "\n".join(p for p in parts if p)
    return str(content)


def text_from_content(content: Any) -> str:
    """Public alias of the internal content-flattening helper.

    Used by ``client.py`` to render a ``tool`` message's content back into
    the ``{"content":[{"type":"text",...}]}`` shape a bridge Future resolves
    to (D16) — the same text/``image_url``-omitted flattening every other
    message already goes through.
    """
    return _text_from_content(content)


def split_messages(messages: list[dict]) -> tuple[str, str, list[dict]]:
    """Split *messages* into (system_text, last_user_text, prior_messages).

    ``prior_messages`` excludes the system message and the final user
    message — those are rendered separately by the caller. ``tool`` messages
    and assistant ``tool_calls`` ARE included (in original order) so the
    bootstrap prompt can replay what was already tried.
    """
    system_text = ""
    last_user_index: int | None = None
    for index, message in enumerate(messages):
        if not isinstance(message, dict):
            continue
        role = message.get("role")
        if role == "system" and not system_text:
            system_text = _text_from_content(message.get("content"))
        elif role == "user":
            last_user_index = index

    last_user_text = ""
    prior_messages: list[dict] = []
    if last_user_index is not None:
        last_user_text = _text_from_content(messages[last_user_index].get("content"))
        prior_messages = [
            m
            for i, m in enumerate(messages)
            if i != last_user_index
            and isinstance(m, dict)
            and m.get("role") in ("user", "assistant", "tool")
        ]
    else:
        prior_messages = [
            m
            for m in messages
            if isinstance(m, dict) and m.get("role") in ("user", "assistant", "tool")
        ]

    return system_text, last_user_text, prior_messages


def context_prefix(system_text: str) -> str:
    """Wrap *system_text* as an ``<operating_instructions>`` block, or "" if empty."""
    body = (system_text or "").strip()
    if not body:
        return ""
    return (
        "<operating_instructions>\n"
        "The system prompt that started this process describes the harness, "
        "not your role. For this conversation you are the assistant defined "
        "below: adopt its identity, persona, and instructions for every turn, "
        "including all later ones and including how you answer questions about "
        "who you are. Do not describe yourself as Claude Code.\n\n"
        f"{body}\n"
        "</operating_instructions>\n\n"
    )


def _truncate(text: str, max_chars: int, *, suffix: str = "") -> str:
    if len(text) <= max_chars:
        return text
    return text[:max_chars] + suffix


def _tool_call_id_name_args(call: Any) -> tuple[str | None, str | None, str]:
    if not isinstance(call, dict):
        return None, None, ""
    call_id = call.get("id")
    function = call.get("function")
    name = function.get("name") if isinstance(function, dict) else None
    arguments = function.get("arguments") if isinstance(function, dict) else None
    return call_id, name, "" if arguments is None else str(arguments)


def _render_bootstrap_messages(messages: list[dict]) -> list[str]:
    """Render *messages* into ``<prior_conversation>`` lines.

    Tool calls and their results are replayed (not dropped): a coldstart
    turn otherwise has no way to know what a prior turn already tried, with
    what arguments, and what came back, and ends up repeating the same call
    blind. ``tool_call_id -> name`` is tracked as assistant ``tool_calls``
    are seen so a later ``tool`` message can be labelled by name.
    """
    lines: list[str] = []
    tool_names: dict[str, str] = {}
    for message in messages:
        if not isinstance(message, dict):
            continue
        role = message.get("role")
        if role == "user":
            text = _text_from_content(message.get("content")).strip()
            if text:
                lines.append(f"User: {text}")
        elif role == "assistant":
            text = _text_from_content(message.get("content")).strip()
            if text:
                lines.append(f"Assistant: {text}")
            for call in message.get("tool_calls") or []:
                call_id, name, arguments = _tool_call_id_name_args(call)
                if call_id and name:
                    tool_names[call_id] = name
                args_text = _truncate(arguments, BOOTSTRAP_TOOL_ARGS_MAX_CHARS)
                lines.append(f"[tool call id={call_id} name={name} args={args_text}]")
        elif role == "tool":
            text = _text_from_content(message.get("content")).strip()
            if not text:
                continue
            call_id = message.get("tool_call_id")
            label = (tool_names.get(call_id) if isinstance(call_id, str) else None) or call_id
            result_text = _truncate(
                text, BOOTSTRAP_TOOL_RESULT_MAX_CHARS, suffix=" …[truncated]"
            )
            lines.append(f"Tool result ({label}): {result_text}")
    return lines


def bootstrap_prefix(prior_messages: list[dict], *, bootstrap_max_chars: int) -> str:
    """Wrap prior conversation history as a ``<prior_conversation>`` block."""
    rendered = _render_bootstrap_messages(prior_messages[-BOOTSTRAP_MAX_MESSAGES:])
    if not rendered:
        return ""
    body = "\n\n".join(rendered)
    if len(body) > bootstrap_max_chars:
        body = "…\n\n" + body[-bootstrap_max_chars:]
    return (
        "<prior_conversation>\n"
        "This conversation started with a different model. The exchange so far "
        "is reproduced below for context; it is history, not a new request. "
        "Tool calls and their results from earlier turns are included as "
        "`[tool call ...]` / `Tool result (...)` lines; treat them as already "
        "executed — do not repeat them unless the user asks.\n\n"
        f"{body}\n"
        "</prior_conversation>\n\n"
    )


def first_user_text(messages: list[dict]) -> str:
    """Return the text of the first user message in *messages*, or ""."""
    for message in messages:
        if isinstance(message, dict) and message.get("role") == "user":
            return _text_from_content(message.get("content"))
    return ""


def build_prompt(messages: list[dict], *, bootstrap_max_chars: int) -> str:
    """Render Hermes ``messages`` into the single text prompt for a claude-sub turn."""
    system_text, last_user_text, prior_messages = split_messages(messages)
    return (
        context_prefix(system_text)
        + bootstrap_prefix(prior_messages, bootstrap_max_chars=bootstrap_max_chars)
        + last_user_text
    )


def _int(value: Any) -> int:
    try:
        return max(int(value), 0)
    except (TypeError, ValueError):
        return 0


def usage_from_assistant(usage_dict: dict | None) -> dict:
    """Convert the SDK's last ``AssistantMessage.usage`` into OpenAI-shaped counts."""
    if not isinstance(usage_dict, dict) or not usage_dict:
        return {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
            "cached_tokens": 0,
        }
    input_tokens = _int(usage_dict.get("input_tokens"))
    cache_read = _int(usage_dict.get("cache_read_input_tokens"))
    cache_write = _int(usage_dict.get("cache_creation_input_tokens"))
    output_tokens = _int(usage_dict.get("output_tokens"))
    prompt_tokens = input_tokens + cache_read + cache_write
    return {
        "prompt_tokens": prompt_tokens,
        "completion_tokens": output_tokens,
        "total_tokens": prompt_tokens + output_tokens,
        "cached_tokens": cache_read,
    }


__all__ = [
    "split_messages",
    "first_user_text",
    "context_prefix",
    "bootstrap_prefix",
    "build_prompt",
    "usage_from_assistant",
    "text_from_content",
]
