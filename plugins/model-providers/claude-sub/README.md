# claude-sub provider plugin

Drives `claude-agent-sdk` directly so Hermes can run against a Claude
Pro/Max/Team subscription without an ACP subprocess. Registers itself as an
`external_process` `ProviderProfile` (see `plugins/model-providers/copilot-acp/`
for the sibling pattern) and supplies its own client via
`ProviderProfile.create_client()` — no core edits required.

## Status: v0.1-A (this card)

- Profile registration, settings, a scrubbed child environment, error
  mapping, message→prompt conversion, and a synchronous one-shot session
  runner.
- The client path is **tool-less and one-shot**: every `chat.completions.create()`
  call spins up a fresh single-turn `SdkSession` (`allowed_tools=[]`,
  `tools=[]`, `max_turns=1`, no MCP servers), waits for the text response, and
  tears the session down. `tools` passed by the caller are accepted but
  ignored.
- No streaming transport, no tool bridging, no session reuse/continuation.
  Those land in a follow-up card (B) on top of this branch.

## Prerequisites

Run `claude auth login` (the bundled or system `claude` CLI) before using
this provider — the SDK resolves that login itself; Hermes never holds,
forwards, or refreshes the credential.

## Configuration

Optional `claude_sub:` block in `config.yaml` (all keys optional):

```yaml
claude_sub:
  start_timeout: 60.0       # seconds to wait for the SDK session to connect
  turn_timeout: 1800.0      # seconds to wait for one turn to complete
  stall_timeout: 300.0      # seconds of silence before a turn is aborted (0 disables)
  orphan_timeout: 120.0     # reserved for card B's open-turn watchdog
  bootstrap_max_chars: 60000  # cap on replayed prior-conversation history
  identity_append: ""       # override the default Hermes identity append
```

## Files

- `__init__.py` — `ClaudeSubProfile` registration.
- `config.py` — `Settings` / `load_settings()`.
- `env.py` — sanitized child environment + subprocess transport.
- `errors.py` — SDK failure → `openai` exception mapping.
- `convert.py` — Hermes messages → SDK prompt, usage accounting.
- `session.py` — synchronous facade over the async SDK client.
- `client.py` — the OpenAI-client-shaped facade (`ClaudeSubClient`).
