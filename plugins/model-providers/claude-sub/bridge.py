"""OpenAI ``tools`` -> in-process SDK MCP bridge (tool-call inversion).

Wraps each of Hermes' OpenAI-shaped function tools as an in-process
``claude_agent_sdk`` MCP tool so the CLI subprocess can call it directly.
Each generated handler hands the call to ``on_call(name, args)`` — a
closure ``client.py`` supplies that returns a ``concurrent.futures.Future``
— and awaits it; ``client.py`` resolves that Future once Hermes core
delivers the real tool result on a later ``create()`` call. This module
never touches Hermes' tool execution itself; it only relays.

The ``PreToolUse`` deny hook is adapted from
``agent/claude_runtime.py:253-284`` (a fork-only module this plugin cannot
import, see AGENTS.md D4) — same pattern, copied rather than shared: deny
every tool call that isn't one of ours, so the CLI's visible-but-inert
built-in tools (kept in context for its billing classifier, see
``session.build_options``) never actually execute.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any, Callable

logger = logging.getLogger(__name__)

BRIDGE_SERVER_NAME = "hermes"
BRIDGE_PREFIX = f"mcp__{BRIDGE_SERVER_NAME}__"

# The claude CLI defers loading MCP tool schemas behind this built-in
# metatool (it lists names only until ``ToolSearch`` is called to fetch a
# tool's full schema). Denying it like every other non-bridge tool leaves
# the model unable to ever see ``mcp__hermes__*`` schemas at all. It is
# safe to allow unconditionally: it only loads schemas and never executes
# anything itself, so it can't bypass the "Hermes owns tool execution"
# contract this hook enforces for everything else.
PASSTHROUGH_TOOLS = frozenset({"ToolSearch"})

# Tools safe to mark read-only for the CLI's own permission UI (irrelevant
# here since PreToolUse allows/denies unconditionally, but the SDK exposes
# the hint to the model regardless).
READ_ONLY = frozenset(
    {
        "read_file",
        "search_files",
        "web_search",
        "web_extract",
        "session_search",
        "skills_list",
        "skill_view",
        "todo",
        "kanban_show",
        "kanban_list",
    }
)


def _make_handler(name: str, on_call: Callable[[str, dict], Any]) -> Callable[[dict], Any]:
    async def _handler(args: dict) -> dict:
        logger.info("claude-sub: bridge handler invoked for tool %s", name)
        fut = on_call(name, args or {})
        result = await asyncio.wrap_future(fut)
        logger.info("claude-sub: bridge handler for tool %s resolved", name)
        return result

    return _handler


def build_bridge(
    tools: list[dict[str, Any]] | None,
    on_call: Callable[[str, dict], Any],
) -> tuple[Any, list[str]]:
    """Wrap Hermes' OpenAI ``tools`` as an in-process SDK MCP server.

    *on_call* is invoked as ``on_call(name, args)`` (the bare Hermes tool
    name, no prefix) and must return a ``concurrent.futures.Future`` whose
    result resolves to ``{"content": [...], "is_error": bool}``.

    Returns ``(mcp_server, allowed_tool_names)`` — each allowed name already
    carries the ``mcp__hermes__`` prefix the SDK exposes registered tools
    under, ready to hand straight to ``ClaudeAgentOptions.allowed_tools``.
    """
    from claude_agent_sdk import ToolAnnotations, create_sdk_mcp_server
    from claude_agent_sdk import tool as sdk_tool

    sdk_tools = []
    allowed_tool_names = []
    for entry in tools or []:
        if not isinstance(entry, dict):
            continue
        fn = entry.get("function") or {}
        if not isinstance(fn, dict):
            continue
        name = fn.get("name")
        if not isinstance(name, str) or not name.strip():
            continue
        name = name.strip()
        description = fn.get("description") or ""
        parameters = fn.get("parameters") or {"type": "object", "properties": {}}
        annotations = ToolAnnotations(readOnlyHint=name in READ_ONLY)
        decorated = sdk_tool(name, description, parameters, annotations=annotations)(
            _make_handler(name, on_call)
        )
        sdk_tools.append(decorated)
        allowed_tool_names.append(f"{BRIDGE_PREFIX}{name}")

    server = create_sdk_mcp_server(name=BRIDGE_SERVER_NAME, version="0.1.0", tools=sdk_tools)
    return server, allowed_tool_names


async def _deny_non_bridge_tool(hook_input: Any, _tool_use_id: Any, _context: Any) -> dict:
    name = str((hook_input or {}).get("tool_name") or "")
    if name.startswith(BRIDGE_PREFIX) or name in PASSTHROUGH_TOOLS:
        return {}
    return {
        "hookSpecificOutput": {
            "hookEventName": "PreToolUse",
            "permissionDecision": "deny",
            "permissionDecisionReason": (
                f"Hermes owns tool execution. Use the {BRIDGE_PREFIX}* "
                f"equivalent of {name or 'this tool'} instead."
            ),
        }
    }


def build_pretooluse_hooks() -> dict[str, Any]:
    """``PreToolUse`` hook dict that denies every non-bridge tool call."""
    from claude_agent_sdk import HookMatcher

    return {"PreToolUse": [HookMatcher(matcher=None, hooks=[_deny_non_bridge_tool])]}


__all__ = [
    "BRIDGE_SERVER_NAME",
    "BRIDGE_PREFIX",
    "PASSTHROUGH_TOOLS",
    "READ_ONLY",
    "build_bridge",
    "build_pretooluse_hooks",
]
