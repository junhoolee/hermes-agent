# claude-sub provider plugin

Drives `claude-agent-sdk` directly so Hermes can run against a Claude
Pro/Max/Team subscription without an ACP subprocess. Registers itself as an
`external_process` `ProviderProfile` (see `plugins/model-providers/copilot-acp/`
for the sibling pattern) and supplies its own client via
`ProviderProfile.create_client()` — no core edits required.

## Status: v0.1-F

- v0.1-A (profile registration, settings, scrubbed child environment, error
  mapping, message→prompt conversion, synchronous one-shot session runner) plus:
- **Bootstrap tool history** (v0.1-F, `convert.py`): a coldstart's
  `<prior_conversation>` replay used to drop every `role: "tool"` message and
  assistant `tool_calls` entirely, so a model resuming mid-conversation (a
  new/mismatched turn, `idle_session_ttl` expiry, `idle_session_ttl: 0`, or a
  provider fallback) had no way to see what a prior turn already tried, with
  what arguments, or what came back — the reported symptom was the model
  blindly repeating the same tool call. `split_messages()` now keeps `tool`
  messages (and assistant `tool_calls`) in `prior_messages`, in original
  order, and `bootstrap_prefix()` renders each assistant tool call as its own
  `[tool call id=<id> name=<name> args=<arguments, truncated to
  `BOOTSTRAP_TOOL_ARGS_MAX_CHARS`=500 chars>]` line, and each `tool` result as
  `Tool result (<name>): <text, truncated to
  `BOOTSTRAP_TOOL_RESULT_MAX_CHARS`=2000 chars, marked "…[truncated]">` — the
  name is resolved by looking up `tool_call_id` against the assistant
  `tool_calls` seen earlier in the same replay, falling back to the raw id if
  unresolved. The `<prior_conversation>` preamble now tells the model these
  lines are already-executed history, not a request to repeat them.
- **Tool-call inversion** (`bridge.py`): Hermes' OpenAI `tools` are wrapped as
  an in-process `claude_agent_sdk` MCP server (`mcp__hermes__<name>`). A
  `PreToolUse` hook denies every non-bridge tool call (except `ToolSearch`,
  the CLI's own deferred-schema-load metatool — see "Bridge id binding"
  below) so the CLI's own built-in tools (kept in context only for its
  billing classifier) never actually run. `client.py` pauses the SDK turn at
  each tool-call boundary, returns an OpenAI-shaped `tool_calls` completion,
  and resumes the *same* SDK turn (no new `query()`) once Hermes core's
  follow-up `create()` call delivers the matching `tool` result messages.
- **Bridge id binding, no wait** (v0.1-D): a `ToolUseBlock` becomes a
  `tool_calls` response the *instant* the projector sees it — there is no
  poll, no timeout, no wait on the matching bridge handler of any kind.
  Earlier revisions (see git history) had the projector block on a short
  poll for the SDK's MCP handler to register before returning; that design
  was unsound at its root, not just too short a timeout — a block whose
  handler never actually runs isn't a failure case, it's how the CLI's own
  `ToolSearch`-driven internal tool resolution normally behaves (see
  `turn.cli_resolved` below), and every such block eventually wedged the
  poll, got discarded, and forced the whole CLI subprocess to restart from a
  cold start — the reported symptom was a 20-minute loop of repeated cold
  starts. The SDK's MCP tool-call handler thread and the drain thread's
  tool-use-block projection still run concurrently with no ordering
  guarantee between them, and now so does the `PreToolUse` hook that fires
  before the handler; `client.py` reconciles all three purely by matching on
  `(name, args)` — never by waiting for one to catch up with the other:
  - `turn.hook_seen` — `PreToolUse` observations (`bridge.build_hooks`'s
    `on_pre_tool_use`), consumed first by `on_call` if present.
  - `turn.outstanding` — every call id sent to Hermes core as `tool_calls`,
    consulted by `on_call` next (exact `(name, args)` match, falling back to
    a `name`-only match for disambiguation-not-required cases).
  - `turn.unbound` — a handler that reached `on_call` before its block was
    ever projected parks here; `_note_tool_blocks` claims it the moment the
    block shows up.
  - `turn.results` — if Hermes core's continuation delivers a tool result
    before any handler ever ran for that id (a `_note_tool_blocks`-only
    block with no handler in flight yet), the result is stashed here and
    handed straight to the handler's Future, already resolved, the moment
    `on_call` eventually catches up.
  - `turn.cli_resolved` — `PostToolUse`/`PostToolUseFailure` observations
    (`on_post_tool_use`) mark a call id as resolved by the CLI itself when
    no handler ever ran for it; `_note_tool_blocks` drops any further block
    for that id instead of sending it to Hermes core.
  All of this is under `turn.lock`; none of it blocks. The claude CLI also
  defers loading MCP tool schemas behind its own `ToolSearch` metatool;
  denying it like every other non-bridge tool left the model unable to ever
  discover `mcp__hermes__*` schemas at all, so it's explicitly passed
  through (it only loads schemas and never executes anything, so this
  doesn't weaken the deny hook's "Hermes owns tool execution" contract).
  To read the reconciliation from `agent.log`: `tool_calls ready ids=[...]`
  logs the moment a block was sent (no wait happened first — timestamps
  should be effectively simultaneous with the model's own tool-call
  message), `PostToolUse id=... handled=... failed=... response=...` logs
  every CLI-side tool-lifecycle observation (compare its `handled` flag
  against whether a later `bridge handler invoked/resolved` line for the
  same id shows up), and `continuation result for ... stashed; handler not
  invoked yet` marks the (b)/(e) stash path above actually firing.
- **Real streaming**: `create(stream=True)` returns a generator of
  OpenAI-shaped delta chunks (text, thinking/reasoning, then a `tool_calls`
  or `stop` chunk followed by a usage chunk), built off the SDK's
  `StreamEvent` partial messages rather than buffering a full turn first.
- **Watchdogs**: a silence (`stall_timeout`) watchdog that's exempt while a
  bridge tool call is in flight, and an open-turn (`orphan_timeout`) watchdog
  that interrupts and tears down a turn paused on `tool_calls` if Hermes core
  never sends the continuation (fallback switch, upstream error, etc.).
- Still one-session-per-turn for a genuinely *new* user turn (system/model/
  reasoning/tools changed, or the history doesn't match a warm session's own
  record) — see "Known limitations".
- **Warm session retention** (v0.1-E): the whole point of this plugin is to
  avoid the cold-start-per-turn tax of a subprocess-based provider (~7-12s
  CLI spawn plus a full bootstrapped-history reprompt, ~4M tokens
  re-transmitted). After a turn reaches `finish_reason="stop"`, the
  underlying `SdkSession` (its CLI subprocess and event-loop thread) is kept
  alive — `idle` — instead of closed, *if and only if* the caller gave an
  explicit `extra_body["hermes_session_id"]` (the aux one-shot client never
  does, so it keeps closing immediately — unaffected). An idle session is
  reclaimed by a `claude_sub.idle_session_ttl` timer (default 30 minutes; `0`
  disables warm retention entirely, reverting to always-close).
  - A follow-up `create()` is treated as a **warm follow-up** — same
    `SdkSession`, a plain `query()` with *only the new user text* as the
    prompt, no `<operating_instructions>`/`<prior_conversation>` re-wrapping
    (the CLI already holds that context; see `client.py`'s
    `_is_warm_followup`) — only when *every* one of these holds:
    the caller's `messages` is exactly this session's known history plus new
    user text (the assistant reply on record must match verbatim — a
    compaction/history rewrite invalidates it), and the `system` text,
    `model`, `reasoning_effort`, and tool set are all unchanged. Any mismatch
    is treated as a new conversation: the idle session is discarded (closed)
    and a fresh one opened with the full bootstrapped prompt, exactly like
    v0.1-D.
  - The v0.1-D tool-call inversion contract (pause on `tool_calls`, resume
    via `continue_turn()` on the matching continuation) is unaffected by
    warm retention — it operates purely on the in-flight `_Turn`, whether
    that turn was opened cold or via a warm follow-up.
  - Idle-timer bookkeeping: a follow-up arriving before the TTL expires
    cancels the timer and resumes the same session. The lookup, the warm
    decision, and claiming the turn (`state` -> `"open"`) all happen inside
    one `_turns_lock` critical section in `_create_chat_completion`, and
    `_arm_idle_timer`'s callback re-checks the same `_Turn` identity and
    `state == "idle"` under that same lock before popping it and closing —
    so the two sides can never interleave: whichever gets the lock first
    wins outright, either claiming the turn (the timer then sees
    `state != "idle"` and no-ops) or closing it (the create() call then finds
    the turn already gone and falls back to a cold restart). The CLI never
    ends up serving a `run_turn()`/`continue_turn()` call against a session
    the timer already closed.
  - `ClaudeSubClient.close()` cancels every idle timer and closes every idle
    session, same as any other open turn.

## Prerequisites

Run `claude auth login` (the bundled or system `claude` CLI) before using
this provider — the SDK resolves that login itself; Hermes never holds,
forwards, or refreshes the credential.

## Installation

This plugin currently ships out of tree, from wherever this repo checkout
lives. To install it into a live `$HERMES_HOME` instead, copy the directory
over:

```bash
cp -R plugins/model-providers/claude-sub "$HERMES_HOME/plugins/model-providers/claude-sub"
```

## Configuration

Optional `claude_sub:` block in `config.yaml` (all keys optional):

```yaml
claude_sub:
  start_timeout: 60.0       # seconds to wait for the SDK session to connect
  turn_timeout: 1800.0      # seconds to wait for one turn to complete
  stall_timeout: 300.0      # seconds of silence before a turn is aborted (0 disables)
  orphan_timeout: 120.0     # seconds to wait for the continuation create() after a
                             # tool_calls response before interrupting the open turn
  bootstrap_max_chars: 60000  # cap on replayed prior-conversation history
  identity_append: ""       # override the default Hermes identity append
  idle_session_ttl: 1800.0  # seconds to keep a stopped session's CLI subprocess
                             # warm for a possible follow-up turn (0 disables —
                             # always close immediately on stop, the v0.1-D behavior)
```

Non-streaming callers (e.g. the auxiliary client) should also set the core
provider-level stale-timeout knob, since a bridged turn can legitimately sit
quiet-to-the-wire while a Hermes tool call runs and the read side just waits
for the next chunk:

```yaml
providers:
  claude-sub:
    stale_timeout_seconds: 1800
```

## Known limitations (v0.1)

- No image input (`supports_vision=False`).
- No `resume`/session-store integration across process restarts — this
  plugin never resumes a prior transcript after Hermes itself restarts. Within
  one running process, a session can be kept warm across turns (v0.1-E, see
  above) as long as the caller passes a stable `hermes_session_id` and the
  history/model/tools stay exactly as that session last saw them; anything
  else (a genuinely new conversation, a history mismatch, `idle_session_ttl`
  expiry, or `idle_session_ttl: 0`) falls back to a fresh SDK session that
  replays the bootstrapped conversation as its prompt — as of v0.1-F that
  replay includes prior `tool_calls`/`tool` history (see above), so this
  fallback path no longer hides earlier tool activity from the model; it is
  still a text replay, not a live tool round-trip, so per-message truncation
  (500/2000 chars) and the 200-message/`bootstrap_max_chars` caps still apply.
- Interrupting a turn that's mid bridge-tool-call has the same latency as any
  other interrupt — the SDK doesn't cancel an in-flight tool result wait any
  faster than a normal generation.
- One SDK session per new (cold) turn, not per tool round-trip: a
  multi-tool-call turn (`tool_calls` → continuation → `tool_calls` →
  continuation → ...) stays on the same session/turn until the final `stop`;
  only a genuinely new or mismatched user turn opens a new one.

## E2E reproduction

```bash
PY=/Users/hermes/.hermes/hermes-agent/.venv/bin/python
HERMES_HOME=/tmp/hermes-claude-sub-v01 $PY -m hermes_cli.main \
  -z "hello, reply in one short sentence" --provider claude-sub -m claude-sonnet-5

printf 'The secret word is PELICAN-7342.\n' > /tmp/hermes-claude-sub-v01/e2e/secret.txt
HERMES_HOME=/tmp/hermes-claude-sub-v01 $PY -m hermes_cli.main \
  -z "Use the read_file tool to read /tmp/hermes-claude-sub-v01/e2e/secret.txt and reply with only the secret word it contains." \
  --provider claude-sub -m claude-sonnet-5 -t file
```

Notes on reproducing the above:

- `-z`/`--oneshot` calls `logging.disable(logging.CRITICAL)`
  (`hermes_cli/oneshot.py`) before running, so none of this plugin's
  `logger.info(...)` lines (or any other logger) reach `agent.log` in this
  mode — that's a oneshot-mode-wide behavior, not specific to claude-sub, and
  out of scope here to change. To see the bridge's own log lines (`create()
  ... continuation=...`, `bridge handler invoked/resolved`, `continuation
  resolving pending tool call ...`, `turn finished`), drive `ClaudeSubClient`
  directly outside oneshot mode instead — see the probe script this
  plugin's own test/verification tooling uses for exactly that.
- The two-message form above (bare system/user text, no surrounding Hermes
  system prompt) is more prompt-injection-suspicious to the underlying model
  than it looks: at that bare level the `<operating_instructions>` wrapper
  (`convert.py`'s `context_prefix`) has no other legitimate context around
  it, and asking it to "reply with only the secret word" a file contains
  reads like a textbook exfiltration-under-injection test payload. In
  isolated testing this combination reproducibly triggered a Claude safety
  refusal (`finish_reason="stop"`, no tool call at all) even though the
  bridge plumbing itself was working — dropping just the "reply with only
  the secret word it contains" phrasing (e.g. "...and tell me what it
  says.") was enough to get a normal tool call every time. The full `-z`
  oneshot path above does **not** hit this refusal with the literal
  wording — Hermes' real (much larger, self-consistent) system prompt gives
  the model enough legitimate context that it treats the embedded
  instructions as expected rather than injected. This is a model-behavior
  property of the exact wording, not a regression in this plugin; if a
  future model revision starts refusing the literal `-z` command too,
  rephrase the ask rather than assuming the bridge broke.

## Files

- `__init__.py` — `ClaudeSubProfile` registration.
- `config.py` — `Settings` / `load_settings()`.
- `env.py` — sanitized child environment + subprocess transport.
- `errors.py` — SDK failure → `openai` exception mapping.
- `convert.py` — Hermes messages → SDK prompt, usage accounting.
- `session.py` — synchronous facade over the async SDK client (pause/continue).
- `bridge.py` — OpenAI `tools` → in-process SDK MCP server (tool-call inversion).
- `client.py` — the OpenAI-client-shaped facade (`ClaudeSubClient`).
