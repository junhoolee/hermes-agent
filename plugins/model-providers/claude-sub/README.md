# claude-sub provider plugin

Drives `claude-agent-sdk` directly so Hermes can run against a Claude
Pro/Max/Team subscription without an ACP subprocess. Registers itself as an
`external_process` `ProviderProfile` (see `plugins/model-providers/copilot-acp/`
for the sibling pattern) and supplies its own client via
`ProviderProfile.create_client()` — no core edits required.

## Status: v0.1-C

- v0.1-A (profile registration, settings, scrubbed child environment, error
  mapping, message→prompt conversion, synchronous one-shot session runner) plus:
- **Tool-call inversion** (`bridge.py`): Hermes' OpenAI `tools` are wrapped as
  an in-process `claude_agent_sdk` MCP server (`mcp__hermes__<name>`). A
  `PreToolUse` hook denies every non-bridge tool call (except `ToolSearch`,
  the CLI's own deferred-schema-load metatool — see "Bridge id binding"
  below) so the CLI's own built-in tools (kept in context only for its
  billing classifier) never actually run. `client.py` pauses the SDK turn at
  each tool-call boundary, returns an OpenAI-shaped `tool_calls` completion,
  and resumes the *same* SDK turn (no new `query()`) once Hermes core's
  follow-up `create()` call delivers the matching `tool` result messages.
- **Bridge id binding** (v0.1-C): the SDK's MCP tool-call handler and the
  drain thread's tool-use-block projection run on different threads with no
  ordering guarantee — the handler can fire before or after its matching
  block is projected. `client.py`'s `on_call`/`_register_expected` reconcile
  whichever side arrives second against the other's parked state
  (`turn.unbound` / `turn.expected_ids`) under `turn.lock`, replacing the
  earlier "handler always arrives after its block" assumption (which
  generated a throwaway id on mismatch and caused a mismatch-loop with
  Hermes core re-issuing the same tool call). `_wait_for_pending` now raises
  `BridgeBindTimeout` after `BRIDGE_BIND_TIMEOUT` (10s) instead of silently
  returning, so a genuinely wedged handler surfaces as a mapped 503 instead
  of an id mismatch. The claude CLI also defers loading MCP tool schemas
  behind its own `ToolSearch` metatool; denying it like every other
  non-bridge tool left the model unable to ever discover `mcp__hermes__*`
  schemas at all, so it's now explicitly passed through (it only loads
  schemas and never executes anything, so this doesn't weaken the deny
  hook's "Hermes owns tool execution" contract).
- **Real streaming**: `create(stream=True)` returns a generator of
  OpenAI-shaped delta chunks (text, thinking/reasoning, then a `tool_calls`
  or `stop` chunk followed by a usage chunk), built off the SDK's
  `StreamEvent` partial messages rather than buffering a full turn first.
- **Watchdogs**: a silence (`stall_timeout`) watchdog that's exempt while a
  bridge tool call is in flight, and an open-turn (`orphan_timeout`) watchdog
  that interrupts and tears down a turn paused on `tool_calls` if Hermes core
  never sends the continuation (fallback switch, upstream error, etc.).
- Still one-session-per-turn: v0.1 opens a brand new `SdkSession` for every
  fresh user turn (not every tool round-trip) and replays the whole
  bootstrapped conversation as the prompt — see "Known limitations".

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
- No `resume`/session-store integration — every fresh user turn starts a new
  SDK session and replays the bootstrapped conversation as its prompt.
  Prompt-cache reuse across turns is a known v0.1 trade-off, not a bug.
- Interrupting a turn that's mid bridge-tool-call has the same latency as any
  other interrupt — the SDK doesn't cancel an in-flight tool result wait any
  faster than a normal generation.
- One SDK session per new turn, not per tool round-trip: a multi-tool-call
  turn (`tool_calls` → continuation → `tool_calls` → continuation → ...) stays
  on the same session/turn until the final `stop`; only a genuinely new user
  message opens a new one.

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
