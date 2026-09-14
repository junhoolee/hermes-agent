"""Message <-> prompt conversion and usage accounting for claude-sub.

The claude-agent-sdk session speaks one prompt per turn, not an OpenAI
``messages`` list. That prompt is usually plain text — rendered from the
Hermes conversation, including a coldstart's prior ``tool_calls``/``tool``
history, so a model resuming mid-conversation can see what was already
tried, with what arguments, and what came back, instead of repeating the
same call blind — but when the last (or, for a warm follow-up, newest) user
message carries an ``image_url``/``input_image`` part, it becomes a
``StreamPrompt``: an async-iterable of one stream-json user frame carrying
an Anthropic image block, so the SDK sends it over stdin instead of losing
it to a ``"[image omitted]"`` placeholder. This module also turns the SDK's
per-call usage dict into OpenAI-shaped token counts.
"""

from __future__ import annotations

import re
from typing import Any

BOOTSTRAP_MAX_MESSAGES = 200
BOOTSTRAP_TOOL_ARGS_MAX_CHARS = 500
BOOTSTRAP_TOOL_RESULT_MAX_CHARS = 2000

IMAGE_PART_TYPES = frozenset({"image_url", "input_image"})

_DATA_URL_RE = re.compile(
    r"^data:(?P<mime>[\w.+-]+/[\w.+-]+)?(?P<params>;[^,]*)?,(?P<data>.*)$", re.DOTALL
)


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


def _image_part_block(part: dict) -> dict:
    """Return the stream block a single image_url/input_image *part* becomes.

    A ``data:`` URL becomes a base64 image block, an ``http(s)://`` URL
    becomes a url image block, and anything else that claims to be an image
    (a non-base64 data URL, an empty url, a bare file path, ...) becomes a
    text block noting the image could not be encoded — it is never silently
    dropped.
    """
    image_url = part.get("image_url")
    url = image_url.get("url") if isinstance(image_url, dict) else image_url
    if isinstance(url, str) and url:
        if url.startswith("http://") or url.startswith("https://"):
            return {"type": "image", "source": {"type": "url", "url": url}}
        match = _DATA_URL_RE.match(url) if url.startswith("data:") else None
        if match:
            params = match.group("params") or ""
            data = match.group("data") or ""
            if "base64" in params and data:
                return {
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": match.group("mime") or "image/png",
                        "data": data.strip(),
                    },
                }
    return {"type": "text", "text": "[an image was attached but could not be encoded]"}


def image_blocks_from_content(content: Any) -> list[dict]:
    """Extract Anthropic image blocks from an OpenAI-shaped multipart *content*.

    Only ``IMAGE_PART_TYPES`` parts are considered; see ``_image_part_block``
    for how each one is encoded (or noted as unencodable).
    """
    if not isinstance(content, list):
        return []
    return [
        _image_part_block(part)
        for part in content
        if isinstance(part, dict) and part.get("type") in IMAGE_PART_TYPES
    ]


def _text_excluding_encoded_images(content: Any) -> str:
    """Like ``_text_from_content`` but for a message whose image blocks ride
    along in the same ``StreamPrompt``: an image part that was actually
    turned into a real ``image`` block is omitted from the text entirely
    (the block right next to it already carries the image, so asserting
    "[image omitted]" would contradict it). A part that failed to encode
    still gets the placeholder, since no image block represents it.
    """
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
            elif part_type in IMAGE_PART_TYPES and _image_part_block(part).get("type") != "image":
                parts.append("[image omitted]")
        return "\n".join(p for p in parts if p)
    return str(content)


class StreamPrompt:
    """An async-iterable stream-json user frame carrying text + image blocks.

    ``ClaudeSDKClient.query()`` accepts either a plain string or an
    ``AsyncIterable[dict]`` of raw stream-json frames; this is the latter,
    used only when the outgoing user turn includes at least one image
    block. Each ``__aiter__()`` call returns a fresh async generator that
    yields a brand-new frame dict — the SDK mutates the frame it receives
    (filling in ``session_id``), so reusing the same dict/list across a
    retried send would leak that mutation into the resend.
    """

    def __init__(self, blocks: list[dict]) -> None:
        self.blocks = list(blocks)

    def __aiter__(self):
        return self._frames()

    async def _frames(self):
        yield {
            "type": "user",
            "message": {
                "role": "user",
                "content": [dict(block) for block in self.blocks],
            },
            "parent_tool_use_id": None,
        }

    @property
    def text(self) -> str:
        """Concatenated text blocks, for logging/debugging only."""
        return "\n".join(
            str(block.get("text") or "")
            for block in self.blocks
            if isinstance(block, dict) and block.get("type") == "text"
        )

    @property
    def image_count(self) -> int:
        return sum(
            1 for block in self.blocks if isinstance(block, dict) and block.get("type") == "image"
        )

    def __repr__(self) -> str:
        return f"StreamPrompt(text_chars={len(self.text)}, images={self.image_count})"


def make_prompt(text: str, extra_blocks: list[dict]) -> "str | StreamPrompt":
    """Combine *text* with *extra_blocks*, returning a str unless an image is present."""
    image_blocks = [
        block for block in extra_blocks if isinstance(block, dict) and block.get("type") == "image"
    ]
    if not image_blocks:
        notes = [
            str(block.get("text") or "")
            for block in extra_blocks
            if isinstance(block, dict) and block.get("type") == "text"
        ]
        if not notes:
            return text
        return "\n".join(([text] if text else []) + notes)
    blocks: list[dict] = ([{"type": "text", "text": text}] if text else []) + list(extra_blocks)
    return StreamPrompt(blocks)


def build_prompt(messages: list[dict], *, bootstrap_max_chars: int) -> "str | StreamPrompt":
    """Render Hermes ``messages`` into the prompt for a claude-sub turn.

    The text portion is unchanged from before, UNLESS the last user message
    carries at least one image part that was actually encoded into a real
    image block — in that case its text is rendered with the "[image
    omitted]" placeholder for that part dropped (the image block right next
    to it already carries it), and the result is a ``StreamPrompt`` instead
    of a plain string.
    """
    system_text, last_user_text, prior_messages = split_messages(messages)
    last_user_content = None
    for message in messages:
        if isinstance(message, dict) and message.get("role") == "user":
            last_user_content = message.get("content")
    image_blocks = image_blocks_from_content(last_user_content)
    if any(block.get("type") == "image" for block in image_blocks):
        last_user_text = _text_excluding_encoded_images(last_user_content)
    text = (
        context_prefix(system_text)
        + bootstrap_prefix(prior_messages, bootstrap_max_chars=bootstrap_max_chars)
        + last_user_text
    )
    return make_prompt(text, image_blocks)


def build_followup_prompt(tail_messages: list[dict]) -> "str | StreamPrompt":
    """Render a warm follow-up's new tail messages into a prompt.

    Text is joined the same way ``client.py``'s warm-followup path always
    has (each message's flattened text, ``"\\n\\n"``-joined) — UNLESS the
    tail carries at least one image part that was actually encoded into a
    real image block, in which case each message's text is rendered with
    the "[image omitted]" placeholder dropped for its encoded part(s). Image
    parts are collected from the tail messages in order and appended as
    image blocks.
    """
    image_blocks: list[dict] = []
    for message in tail_messages:
        if isinstance(message, dict):
            image_blocks.extend(image_blocks_from_content(message.get("content")))
    render_text = (
        _text_excluding_encoded_images
        if any(block.get("type") == "image" for block in image_blocks)
        else _text_from_content
    )
    text = "\n\n".join(render_text(message.get("content")) for message in tail_messages)
    return make_prompt(text, image_blocks)


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
    "build_followup_prompt",
    "make_prompt",
    "image_blocks_from_content",
    "IMAGE_PART_TYPES",
    "StreamPrompt",
    "usage_from_assistant",
    "text_from_content",
]
