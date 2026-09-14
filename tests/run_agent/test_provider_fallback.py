"""Tests for ordered provider fallback chain (salvage of PR #1761).

Extends the single-fallback tests in test_fallback_model.py to cover
the new list-based ``fallback_providers`` config format and chain
advancement through multiple providers.
"""

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from agent import chat_completion_helpers
from agent.error_classifier import FailoverReason
from run_agent import AIAgent, _pool_may_recover_from_rate_limit


def _make_agent(fallback_model=None):
    """Create a minimal AIAgent with optional fallback config."""
    with (
        patch("run_agent.get_tool_definitions", return_value=[]),
        patch("run_agent.check_toolset_requirements", return_value={}),
        patch("run_agent.OpenAI"),
    ):
        agent = AIAgent(
            api_key="test-key",
            base_url="https://openrouter.ai/api/v1",
            quiet_mode=True,
            skip_context_files=True,
            skip_memory=True,
            fallback_model=fallback_model,
        )
        agent.client = MagicMock()
        return agent


def _mock_client(base_url="https://openrouter.ai/api/v1", api_key="fb-key"):
    mock = MagicMock()
    mock.base_url = base_url
    mock.api_key = api_key
    return mock


# ── Chain initialisation ──────────────────────────────────────────────────


class TestFallbackChainInit:
    def test_no_fallback(self):
        agent = _make_agent(fallback_model=None)
        assert agent._fallback_chain == []
        assert agent._fallback_index == 0
        assert agent._fallback_model is None



    def test_invalid_entries_filtered(self):
        fbs = [
            {"provider": "openai", "model": "gpt-4o"},
            {"provider": "", "model": "glm-4.7"},
            {"provider": "zai"},
            "not-a-dict",
        ]
        agent = _make_agent(fallback_model=fbs)
        assert len(agent._fallback_chain) == 1
        assert agent._fallback_chain[0]["provider"] == "openai"


    def test_invalid_dict_no_provider(self):
        agent = _make_agent(fallback_model={"model": "gpt-4o"})
        assert agent._fallback_chain == []


# ── Chain advancement ─────────────────────────────────────────────────────


@pytest.mark.parametrize(
    ("reason", "expected"),
    [
        (FailoverReason.auth, "authentication failed"),
        (FailoverReason.billing, "billing or quota exhausted"),
        (FailoverReason.rate_limit, "rate limit"),
        (FailoverReason.upstream_rate_limit, "upstream model rate limit"),
        (FailoverReason.overloaded, "provider overloaded"),
        (FailoverReason.server_error, "provider server error"),
        (FailoverReason.timeout, "request timeout"),
        (FailoverReason.model_not_found, "model not found"),
        (FailoverReason.unknown, "provider failure"),
    ],
)
def test_fallback_reason_text_is_operator_friendly(reason, expected):
    assert chat_completion_helpers._fallback_reason_text(reason) == expected


def test_fallback_reason_text_defaults_when_reason_is_missing():
    assert chat_completion_helpers._fallback_reason_text(None) == "provider failure"


class TestFallbackChainAdvancement:
    def test_exhausted_returns_false(self):
        agent = _make_agent(fallback_model=None)
        assert agent._try_activate_fallback() is False

    def test_advances_index(self):
        fbs = [
            {"provider": "openai", "model": "gpt-4o"},
            {"provider": "zai", "model": "glm-4.7"},
        ]
        agent = _make_agent(fallback_model=fbs)
        with patch("agent.auxiliary_client.resolve_provider_client",
                    return_value=(_mock_client(), "gpt-4o")):
            assert agent._try_activate_fallback() is True
            assert agent._fallback_index == 1
            assert agent.model == "gpt-4o"
            assert agent._fallback_activated is True

    def test_records_user_visible_switch_with_reason(self):
        agent = _make_agent(
            fallback_model={"provider": "zai", "model": "glm-5.2"},
        )
        agent.model = "gpt-5.6-sol"
        agent.provider = "openai-codex"
        with patch(
            "agent.auxiliary_client.resolve_provider_client",
            return_value=(_mock_client(base_url="https://api.z.ai/v1"), "glm-5.2"),
        ):
            assert agent._try_activate_fallback(FailoverReason.rate_limit) is True

        expected = (
            "⚠️ Model fallback: gpt-5.6-sol via openai-codex unavailable "
            "(rate limit); using glm-5.2 via zai."
        )
        assert agent._pending_fallback_notice == [expected]
        assert agent._retry_status_buffer[-1] == ("status", expected)

    def test_records_sequential_switches_in_order(self):
        agent = _make_agent(
            fallback_model=[
                {"provider": "zai", "model": "glm-5.2"},
                {"provider": "deepseek", "model": "deepseek-v4-flash"},
            ],
        )
        agent.model = "gpt-5.6-sol"
        agent.provider = "openai-codex"
        clients = [
            _mock_client(base_url="https://api.z.ai/v1"),
            _mock_client(base_url="https://api.deepseek.com/v1"),
        ]
        with patch(
            "agent.auxiliary_client.resolve_provider_client",
            side_effect=[(clients[0], "glm-5.2"), (clients[1], "deepseek-v4-flash")],
        ):
            assert agent._try_activate_fallback(FailoverReason.rate_limit) is True
            assert agent._try_activate_fallback(FailoverReason.overloaded) is True

        assert agent._pending_fallback_notice == [
            "⚠️ Model fallback: gpt-5.6-sol via openai-codex unavailable "
            "(rate limit); using glm-5.2 via zai.",
            "⚠️ Model fallback: glm-5.2 via zai unavailable "
            "(provider overloaded); using deepseek-v4-flash via deepseek.",
        ]
    def test_skips_unconfigured_provider_to_next(self):
        """If resolve_provider_client returns None, skip to next in chain."""
        fbs = [
            {"provider": "broken", "model": "nope"},
            {"provider": "openai", "model": "gpt-4o"},
        ]
        agent = _make_agent(fallback_model=fbs)
        with patch("agent.auxiliary_client.resolve_provider_client") as mock_rpc:
            mock_rpc.side_effect = [
                (None, None),                    # broken provider
                (_mock_client(), "gpt-4o"),       # fallback succeeds
            ]
            assert agent._try_activate_fallback() is True
            assert agent.model == "gpt-4o"
            assert agent._fallback_index == 2

    def test_skips_provider_that_raises_to_next(self):
        """If resolve_provider_client raises, skip to next in chain."""
        fbs = [
            {"provider": "broken", "model": "nope"},
            {"provider": "openai", "model": "gpt-4o"},
        ]
        agent = _make_agent(fallback_model=fbs)
        with patch("agent.auxiliary_client.resolve_provider_client") as mock_rpc:
            mock_rpc.side_effect = [
                RuntimeError("auth failed"),
                (_mock_client(), "gpt-4o"),
            ]
            assert agent._try_activate_fallback() is True
            assert agent.model == "gpt-4o"

    def test_resolves_key_env_for_fallback_provider(self):
        fbs = [
            {
                "provider": "custom",
                "model": "fallback-model",
                "base_url": "https://fallback.example/v1",
                "key_env": "MY_FALLBACK_KEY",
            }
        ]
        agent = _make_agent(fallback_model=fbs)
        with (
            patch.dict("os.environ", {"MY_FALLBACK_KEY": "env-secret"}, clear=False),
            patch(
                "agent.auxiliary_client.resolve_provider_client",
                return_value=(
                    _mock_client(
                        base_url="https://fallback.example/v1",
                        api_key="env-secret",
                    ),
                    "fallback-model",
                ),
            ) as mock_rpc,
        ):
            assert agent._try_activate_fallback() is True
            assert mock_rpc.call_args.kwargs["explicit_api_key"] == "env-secret"


    def test_nous_anthropic_fallback_uses_the_messages_wire(self):
        """Portal Claude fallbacks must not stay on chat_completions.

        ``resolve_provider_client`` still returns an OpenAI client for Nous;
        activation has to re-derive api_mode from the model and rebuild the
        Anthropic client — otherwise the turn POSTs /chat/completions.
        """
        portal = "https://inference-api.nousresearch.com/v1"
        fbs = [
            {
                "provider": "nous",
                "model": "anthropic/claude-opus-4.8",
            }
        ]
        agent = _make_agent(fallback_model=fbs)
        rebuilt = {"count": 0}

        def _fake_build(api_key, base_url, timeout=None, **kwargs):
            rebuilt["count"] += 1
            rebuilt["api_key"] = api_key
            rebuilt["base_url"] = base_url
            return MagicMock(name="anthropic-client")

        with (
            patch(
                "agent.chat_completion_helpers._fallback_entry_unavailable_without_network",
                return_value=None,
            ),
            patch(
                "agent.auxiliary_client.resolve_provider_client",
                return_value=(
                    _mock_client(base_url=portal, api_key="portal-jwt"),
                    "anthropic/claude-opus-4.8",
                ),
            ),
            patch(
                "hermes_cli.model_normalize.normalize_model_for_provider",
                side_effect=lambda m, p: m,
            ),
            patch(
                "agent.anthropic_adapter.build_anthropic_client",
                side_effect=_fake_build,
            ),
        ):
            assert agent._try_activate_fallback() is True

        assert agent.api_mode == "anthropic_messages"
        assert agent.provider == "nous"
        assert agent.model == "anthropic/claude-opus-4.8"
        assert agent.client is None
        assert rebuilt["count"] == 1
        assert rebuilt["api_key"] == "portal-jwt"
        assert rebuilt["base_url"] == portal
        assert agent._anthropic_client is not None

    def test_nous_non_anthropic_fallback_stays_on_chat_completions(self):
        portal = "https://inference-api.nousresearch.com/v1"
        fbs = [{"provider": "nous", "model": "hermes-4-405b"}]
        agent = _make_agent(fallback_model=fbs)
        with (
            patch(
                "agent.chat_completion_helpers._fallback_entry_unavailable_without_network",
                return_value=None,
            ),
            patch(
                "agent.auxiliary_client.resolve_provider_client",
                return_value=(
                    _mock_client(base_url=portal, api_key="portal-jwt"),
                    "hermes-4-405b",
                ),
            ),
            patch(
                "hermes_cli.model_normalize.normalize_model_for_provider",
                side_effect=lambda m, p: m,
            ),
            patch(
                "agent.anthropic_adapter.build_anthropic_client",
                side_effect=AssertionError("must not build Anthropic client"),
            ),
        ):
            assert agent._try_activate_fallback() is True

        assert agent.api_mode == "chat_completions"
        assert agent.client is not None


# ── Pool-rotation vs fallback gating (#11314) ────────────────────────────


def _pool(n_entries: int, has_available: bool = True):
    """Make a minimal credential-pool stand-in for rotation-room checks."""
    pool = MagicMock()
    pool.entries.return_value = [MagicMock() for _ in range(n_entries)]
    pool.has_available.return_value = has_available
    return pool


class TestPoolRotationRoom:
    def test_none_pool_returns_false(self):
        assert _pool_may_recover_from_rate_limit(None) is False







# ── Skip-self dedup (#22548) ───────────────────────────────────────────────


class TestFallbackChainDedup:
    """A fallback chain entry that resolves to the current provider/model
    (or the same custom-provider base_url) must be skipped, not retried.
    Otherwise a misconfigured chain or two custom_providers entries pointing
    at the same shim loop the same failure. See issue #22548."""

    def test_skips_entry_matching_current_provider_and_model(self):
        """Chain has [same-as-current, real-fallback]; activate must skip
        the first and use the second."""
        fbs = [
            # First entry == current state. Should be skipped.
            {"provider": "openrouter", "model": "z-ai/glm-4.7"},
            # Second entry: real fallback.
            {"provider": "zai", "model": "glm-4.7"},
        ]
        agent = _make_agent(fallback_model=fbs)
        agent.provider = "openrouter"
        agent.model = "z-ai/glm-4.7"
        agent.base_url = "https://openrouter.ai/api/v1"

        # Stub out resolve_provider_client so we can assert which entry was
        # actually used — return a MagicMock client tagged with the provider.
        called = []
        def _resolve(provider, model=None, raw_codex=False, **kwargs):
            called.append((provider, model))
            return _mock_client(), model
        with patch("agent.auxiliary_client.resolve_provider_client", side_effect=_resolve):
            with patch("hermes_cli.model_normalize.normalize_model_for_provider", side_effect=lambda m, p: m):
                ok = agent._try_activate_fallback()

        assert ok is True
        # The first entry was skipped — only the second reached resolve.
        assert called == [("zai", "glm-4.7")], (
            f"expected fallback to skip same-state entry, got call order: {called}"
        )


    def test_returns_false_when_only_self_matching_entries(self):
        """A chain with only self-matching entries exhausts to False."""
        fbs = [
            {"provider": "openrouter", "model": "z-ai/glm-4.7"},
        ]
        agent = _make_agent(fallback_model=fbs)
        agent.provider = "openrouter"
        agent.model = "z-ai/glm-4.7"
        agent.base_url = "https://openrouter.ai/api/v1"

        with patch("agent.auxiliary_client.resolve_provider_client") as mock_resolve:
            ok = agent._try_activate_fallback()

        assert ok is False
        mock_resolve.assert_not_called()

    def test_allows_xai_api_fallback_from_xai_oauth_same_host_model(self):
        """xai-oauth and xai share api.x.ai but use different credentials.

        A spending-limit 403 on OAuth must still be able to fall over to the
        API-key provider even when both entries use the same model slug and
        base URL.  Blind base_url+model dedup incorrectly skipped that path.
        """
        fbs = [
            {
                "provider": "xai",
                "model": "grok-4.5",
                "base_url": "https://api.x.ai/v1",
            },
        ]
        agent = _make_agent(fallback_model=fbs)
        agent.provider = "xai-oauth"
        agent.model = "grok-4.5"
        agent.base_url = "https://api.x.ai/v1"

        called = []

        def _resolve(provider, model=None, raw_codex=False, **kwargs):
            called.append((provider, model))
            return _mock_client(base_url="https://api.x.ai/v1"), model

        with patch("agent.auxiliary_client.resolve_provider_client", side_effect=_resolve):
            with patch(
                "hermes_cli.model_normalize.normalize_model_for_provider",
                side_effect=lambda m, p: m,
            ):
                ok = agent._try_activate_fallback()

        assert ok is True
        assert called == [("xai", "grok-4.5")]
        assert agent.provider == "xai"
        assert agent.model == "grok-4.5"


# ── extra_body re-resolution on fallback activation (#75091) ─────────────


class TestFallbackExtraBodyReResolution:
    """Fallback activation must re-resolve extra_body key-scoped.

    The old provider's custom_providers-contributed extra_body keys are
    stale on the new backend and must be dropped; caller-provided
    request_overrides keys must survive; the fallback provider's own
    extra_body must be merged in (salvage of #75139).
    """

    OLD_URL = "https://old-llm.example.com/v1"
    FB_URL = "https://fb-llm.example.com/v1"

    def _agent_with_custom_providers(self, caller_extra_body=None):
        agent = _make_agent(
            fallback_model={
                "provider": "custom:fbprov",
                "model": "fb-model",
                "base_url": self.FB_URL,
            },
        )
        agent.provider = "custom"
        agent.model = "old-model"
        agent.base_url = self.OLD_URL
        agent._custom_providers = [
            {
                "name": "oldprov",
                "base_url": self.OLD_URL,
                "extra_body": {"enable_thinking": True, "old_only": 1},
            },
            {
                "provider_key": "fbprov",
                "base_url": self.FB_URL,
                "extra_body": {"top_k": 20},
            },
        ]
        # Simulate the init-time merge: provider extra_body + caller keys
        # (caller wins on conflict — agent_init._merge_custom_provider_extra_body).
        merged = {"enable_thinking": True, "old_only": 1}
        merged.update(caller_extra_body or {})
        agent.request_overrides = {"extra_body": merged}
        return agent

    def _activate(self, agent):
        with patch(
            "agent.auxiliary_client.resolve_provider_client",
            return_value=(_mock_client(base_url=self.FB_URL), "fb-model"),
        ), patch(
            "agent.model_metadata.get_model_context_length",
            return_value=128_000,
        ):
            assert agent._try_activate_fallback() is True

    def test_stale_provider_keys_removed_and_new_provider_merged(self):
        agent = self._agent_with_custom_providers()
        self._activate(agent)
        eb = agent.request_overrides.get("extra_body") or {}
        # Old provider's contributed keys are gone.
        assert "enable_thinking" not in eb
        assert "old_only" not in eb
        # Fallback provider's own extra_body is applied.
        assert eb.get("top_k") == 20

    def test_caller_override_keys_survive_fallback(self):
        agent = self._agent_with_custom_providers(
            caller_extra_body={"reasoning": {"effort": "high"}, "enable_thinking": False},
        )
        self._activate(agent)
        eb = agent.request_overrides.get("extra_body") or {}
        # Pure caller key survives untouched.
        assert eb.get("reasoning") == {"effort": "high"}
        # Caller redefined a key the old provider also set (caller won at
        # init: False != True) — the caller's value must survive key-scoped
        # removal.
        assert eb.get("enable_thinking") is False
        # But the key the old provider alone contributed is dropped.
        assert "old_only" not in eb
        assert eb.get("top_k") == 20

    def test_non_extra_body_overrides_untouched(self):
        agent = self._agent_with_custom_providers()
        agent.request_overrides["temperature"] = 0.2
        self._activate(agent)
        assert agent.request_overrides.get("temperature") == 0.2


# ── Claude subscription (claude_agent_sdk) fallback entries ───────────────


class _FakeRateLimitError(Exception):
    status_code = 429

    def __init__(self):
        super().__init__("Error code: 429 - rate limit exceeded")
        self.response = SimpleNamespace(headers={})
        self.body = {"error": {"message": "rate limit exceeded"}}


class TestClaudeAgentSdkFallback:
    """A `claude-code` chain entry must activate the Claude subscription
    runtime, and leaving it must release the live SDK session — mirrors
    what switch_model() already does for the /model command path."""

    def test_midturn_429_hands_the_same_turn_to_the_sdk_runtime(self):
        """State alone is not enough: the api_mode dispatch in
        run_conversation() runs once, BEFORE the retry loop, so a mid-turn
        fallback activation that lands on claude_agent_sdk must hand the
        rest of THIS turn to the SDK runtime. Without the handoff the retry
        path rebuilds an OpenAI client the swap deliberately removed
        (client=None) and the turn dies on "Failed to recreate closed
        OpenAI client" — Claude never serves a single turn."""
        fbs = [{"provider": "claude-code", "model": "claude-sonnet-5"}]
        agent = _make_agent(fallback_model=fbs)
        agent._api_max_retries = 2

        sdk_turns = []

        def _sdk_turn(**kwargs):
            sdk_turns.append(kwargs)
            return {
                "final_response": "served by the claude sdk runtime",
                "messages": kwargs["messages"],
                "api_calls": 1,
                "completed": True,
            }

        agent._run_claude_agent_sdk_turn = _sdk_turn

        def _always_429(api_kwargs):
            raise _FakeRateLimitError()

        with (
            patch.object(agent, "_interruptible_api_call", side_effect=_always_429),
            patch.object(agent, "_persist_session"),
            patch.object(agent, "_save_trajectory"),
            patch.object(agent, "_cleanup_task_resources"),
            patch("agent.agent_runtime_helpers.time.sleep"),
            patch(
                "hermes_cli.claude_code.subscription_enabled",
                new=lambda config=None: True,
            ),
            patch(
                "agent.chat_completion_helpers._fallback_entry_unavailable_without_network",
                return_value=None,
            ),
            patch(
                "agent.auxiliary_client.resolve_provider_client",
                return_value=(
                    _mock_client(base_url="claude-sdk://subscription", api_key=""),
                    "claude-sonnet-5",
                ),
            ),
            patch(
                "hermes_cli.model_normalize.normalize_model_for_provider",
                side_effect=lambda m, p: m,
            ),
            patch(
                "agent.model_metadata.get_model_context_length",
                return_value=200000,
            ),
        ):
            result = agent.run_conversation("hello")

        assert len(sdk_turns) == 1, (
            "the 429 turn itself must be served by the SDK runtime, "
            f"got result: {result.get('final_response')!r}"
        )
        assert result["final_response"] == "served by the claude sdk runtime"
        assert agent.api_mode == "claude_agent_sdk"

    def test_unknown_declared_api_mode_falls_back_to_derivation(self):
        """A typo'd entry api_mode must not be installed verbatim — it is
        rejected with a warning and the wire is derived instead."""
        fbs = [
            {
                "provider": "zai",
                "model": "glm-4.7",
                "api_mode": "totally-bogus-mode",
            }
        ]
        agent = _make_agent(fallback_model=fbs)
        with (
            patch(
                "agent.chat_completion_helpers._fallback_entry_unavailable_without_network",
                return_value=None,
            ),
            patch(
                "agent.auxiliary_client.resolve_provider_client",
                return_value=(_mock_client(), "glm-4.7"),
            ),
            patch(
                "hermes_cli.model_normalize.normalize_model_for_provider",
                side_effect=lambda m, p: m,
            ),
        ):
            assert agent._try_activate_fallback() is True

        assert agent.api_mode == "chat_completions"

    def test_claude_code_entry_activates_the_sdk_runtime(self):
        """A `claude-code` chain entry must select the claude_agent_sdk
        runtime, not install the auxiliary one-shot adapter as the main
        OpenAI client (which hard-fails every fallback turn on
        "runs with no tools")."""
        fbs = [{"provider": "claude-code", "model": "claude-sonnet-5"}]
        agent = _make_agent(fallback_model=fbs)
        with (
            patch(
                "hermes_cli.claude_code.subscription_enabled",
                new=lambda config=None: True,
            ),
            patch(
                "agent.chat_completion_helpers._fallback_entry_unavailable_without_network",
                return_value=None,
            ),
            patch(
                "agent.auxiliary_client.resolve_provider_client",
                return_value=(
                    _mock_client(base_url="claude-sdk://subscription", api_key=""),
                    "claude-sonnet-5",
                ),
            ),
            patch(
                "hermes_cli.model_normalize.normalize_model_for_provider",
                side_effect=lambda m, p: m,
            ),
        ):
            assert agent._try_activate_fallback() is True

        assert agent.api_mode == "claude_agent_sdk"
        assert agent.provider == "claude-code"
        assert agent.model == "claude-sonnet-5"
        # No HTTP client: the SDK owns auth and transport itself.
        assert agent.client is None
        assert agent._client_kwargs == {}

    def test_entry_declared_api_mode_is_honored(self):
        """`hermes fallback add` writes api_mode into the chain entry —
        activation must honor it instead of re-deriving from the base URL."""
        fbs = [
            {
                "provider": "claude-code",
                "model": "claude-sonnet-5",
                "api_mode": "claude_agent_sdk",
            }
        ]
        agent = _make_agent(fallback_model=fbs)
        with (
            patch(
                "agent.chat_completion_helpers._fallback_entry_unavailable_without_network",
                return_value=None,
            ),
            patch(
                "agent.auxiliary_client.resolve_provider_client",
                return_value=(
                    _mock_client(base_url="claude-sdk://subscription", api_key=""),
                    "claude-sonnet-5",
                ),
            ),
            patch(
                "hermes_cli.model_normalize.normalize_model_for_provider",
                side_effect=lambda m, p: m,
            ),
        ):
            assert agent._try_activate_fallback() is True

        assert agent.api_mode == "claude_agent_sdk"
        assert agent.client is None
        assert agent._client_kwargs == {}

    def test_leaving_the_sdk_runtime_releases_the_session(self):
        """Advancing the chain away from claude-code must release the live
        SDK session (loop thread + Claude Code subprocess) the same way
        switch_model() does, or both leak."""
        fbs = [{"provider": "zai", "model": "glm-4.7"}]
        agent = _make_agent(fallback_model=fbs)
        agent.api_mode = "claude_agent_sdk"
        agent.provider = "claude-code"
        agent.model = "claude-sonnet-5"
        agent.base_url = "claude-sdk://subscription"
        agent._release_claude_agent_sdk_session = MagicMock()
        with (
            patch(
                "agent.chat_completion_helpers._fallback_entry_unavailable_without_network",
                return_value=None,
            ),
            patch(
                "agent.auxiliary_client.resolve_provider_client",
                return_value=(_mock_client(), "glm-4.7"),
            ),
            patch(
                "hermes_cli.model_normalize.normalize_model_for_provider",
                side_effect=lambda m, p: m,
            ),
        ):
            assert agent._try_activate_fallback() is True

        agent._release_claude_agent_sdk_session.assert_called_once()
        assert agent.api_mode == "chat_completions"

    def test_failed_sdk_turn_hands_off_to_the_fallback_chain(self):
        """Reverse of test_midturn_429_hands_the_same_turn_to_the_sdk_runtime:
        the PRIMARY runtime is claude_agent_sdk and the turn fails outright
        (preflight refusal, session-construction error, or an unrecoverable
        run_turn exception — see claude_runtime._failure_result). The
        pre-loop dispatch in run_conversation() must advance the fallback
        chain and let the next entry serve the turn instead of surfacing
        the SDK failure straight to the user."""
        fbs = [{"provider": "zai", "model": "glm-4.7"}]
        agent = _make_agent(fallback_model=fbs)
        agent.api_mode = "claude_agent_sdk"
        agent.provider = "claude-code"
        agent.model = "claude-sonnet-5"
        agent.base_url = "claude-sdk://subscription"
        agent.client = None
        agent._release_claude_agent_sdk_session = MagicMock()

        sdk_turns = []

        def _sdk_turn(**kwargs):
            sdk_turns.append(kwargs)
            return {
                "final_response": "Claude Agent SDK could not start: boom",
                "messages": kwargs["messages"],
                "api_calls": 0,
                "completed": False,
                "partial": True,
                "failed": True,
                "interrupted": False,
                "error": "boom",
            }

        agent._run_claude_agent_sdk_turn = _sdk_turn
        mock_fb_client = _mock_client(base_url="https://open.bigmodel.cn/api/coding/paas/v4")

        def _fake_api_call(api_kwargs):
            msg = SimpleNamespace(content="served by the fallback chain", tool_calls=None)
            choice = SimpleNamespace(message=msg, finish_reason="stop")
            return SimpleNamespace(choices=[choice], model="glm-4.7", usage=None)

        with (
            patch.object(agent, "_interruptible_api_call", side_effect=_fake_api_call),
            patch.object(agent, "_persist_session"),
            patch.object(agent, "_save_trajectory"),
            patch.object(agent, "_cleanup_task_resources"),
            patch(
                "agent.chat_completion_helpers._fallback_entry_unavailable_without_network",
                return_value=None,
            ),
            patch(
                "agent.auxiliary_client.resolve_provider_client",
                return_value=(mock_fb_client, "glm-4.7"),
            ),
            patch(
                "hermes_cli.model_normalize.normalize_model_for_provider",
                side_effect=lambda m, p: m,
            ),
            patch(
                "agent.model_metadata.get_model_context_length",
                return_value=200000,
            ),
        ):
            result = agent.run_conversation("hello")

        assert len(sdk_turns) == 1, (
            "the SDK runtime must be tried exactly once before falling "
            f"back, got result: {result.get('final_response')!r}"
        )
        agent._release_claude_agent_sdk_session.assert_called_once()
        assert agent.api_mode == "chat_completions"
        assert agent.provider == "zai"
        assert agent.model == "glm-4.7"
        assert result["completed"] is True
        assert result["final_response"] == "served by the fallback chain"

    def test_sdk_turn_failure_with_exhausted_chain_surfaces_the_failure(self):
        """No fallback chain configured — the SDK failure must be returned
        as-is, not swallowed."""
        agent = _make_agent(fallback_model=None)
        agent.api_mode = "claude_agent_sdk"
        agent.provider = "claude-code"
        agent.model = "claude-sonnet-5"
        agent.base_url = "claude-sdk://subscription"
        agent.client = None

        def _sdk_turn(**kwargs):
            return {
                "final_response": "Claude Agent SDK could not start: boom",
                "messages": kwargs["messages"],
                "api_calls": 0,
                "completed": False,
                "partial": True,
                "failed": True,
                "interrupted": False,
                "error": "boom",
            }

        agent._run_claude_agent_sdk_turn = _sdk_turn

        with (
            patch.object(agent, "_persist_session"),
            patch.object(agent, "_save_trajectory"),
            patch.object(agent, "_cleanup_task_resources"),
        ):
            result = agent.run_conversation("hello")

        assert result["failed"] is True
        assert agent.api_mode == "claude_agent_sdk"

    def test_interrupted_sdk_turn_is_not_handed_off(self):
        """A user-requested interrupt carries failed=True too, but it is
        not a runtime failure — it must be returned as-is and must not
        burn a fallback-chain slot."""
        fbs = [{"provider": "zai", "model": "glm-4.7"}]
        agent = _make_agent(fallback_model=fbs)
        agent.api_mode = "claude_agent_sdk"
        agent.provider = "claude-code"
        agent.model = "claude-sonnet-5"
        agent.base_url = "claude-sdk://subscription"
        agent.client = None

        calls = {"n": 0}

        def _sdk_turn(**kwargs):
            calls["n"] += 1
            return {
                "final_response": "",
                "messages": kwargs["messages"],
                "api_calls": 0,
                "completed": False,
                "partial": True,
                "failed": True,
                "interrupted": True,
                "error": None,
            }

        agent._run_claude_agent_sdk_turn = _sdk_turn

        with (
            patch.object(agent, "_persist_session"),
            patch.object(agent, "_save_trajectory"),
            patch.object(agent, "_cleanup_task_resources"),
        ):
            result = agent.run_conversation("hello")

        assert calls["n"] == 1
        assert result["interrupted"] is True
        assert agent.api_mode == "claude_agent_sdk"

    def test_a_session_limit_is_error_result_hands_the_turn_to_openai_codex(self):
        """End-to-end reproduction of the observed outage. The Claude Code
        CLI reports a session limit not as an exception but as an ordinary
        ``ResultMessage(is_error=True, result=<limit text>)`` in the normal
        stream. Driven through the REAL ``run_claude_agent_sdk_turn`` (only
        the SDK session itself is stubbed), the pre-loop dispatch must see
        ``failed: True``, advance the chain to the configured
        openai-codex/gpt-5.6-sol entry, serve the SAME turn over
        codex_responses, and keep the limit text out of the transcript so
        role alternation still holds for the fallback provider."""
        from dataclasses import dataclass

        from agent import claude_runtime

        limit_text = "You've hit your session limit · resets 6pm (Asia/Seoul)"

        # The projector dispatches on the class name, so the stand-in must be
        # called ResultMessage exactly like the SDK type.
        @dataclass
        class ResultMessage:
            subtype: str = "success"
            session_id: str = "sdk-session-1"
            result: str | None = None
            usage: dict | None = None
            total_cost_usd: float | None = None
            terminal_reason: str | None = None
            is_error: bool = False
            errors: list | None = None

        class _LimitSession:
            def __init__(self):
                self.closed = False
                self.prompts = []

            def run_turn(
                self, prompt, *, on_message, timeout=None, stall_timeout=None,
                stall_exempt=None,
            ):
                self.prompts.append(prompt)
                on_message(ResultMessage(is_error=True, result=limit_text))
                return 1

            def note_session_id(self, session_id):
                pass

            def request_interrupt_nowait(self):
                return True

            def close(self):
                self.closed = True

        fbs = [
            {
                "provider": "openai-codex",
                "model": "gpt-5.6-sol",
                "base_url": "https://chatgpt.com/backend-api/codex",
            }
        ]
        agent = _make_agent(fallback_model=fbs)
        agent.api_mode = "claude_agent_sdk"
        agent.provider = "claude-code"
        agent.model = "claude-sonnet-5"
        agent.base_url = "claude-sdk://subscription"
        agent.client = None
        # Billing already proven for this session: no throwaway CLI spawn.
        agent._claude_billing_refusal = None
        agent._claude_session = session = _LimitSession()

        codex_calls = []

        def _fake_codex_call(api_kwargs):
            codex_calls.append(api_kwargs)
            return SimpleNamespace(
                output=[
                    SimpleNamespace(
                        type="message",
                        content=[
                            SimpleNamespace(type="output_text", text="served by codex")
                        ],
                    )
                ],
                usage=SimpleNamespace(input_tokens=5, output_tokens=3, total_tokens=8),
                status="completed",
                model="gpt-5.6-sol",
            )

        mock_fb_client = _mock_client(
            base_url="https://chatgpt.com/backend-api/codex", api_key="codex-token"
        )

        with (
            patch.object(claude_runtime, "claude_runtime_preflight", return_value=None),
            patch.object(claude_runtime, "_ensure_session", return_value=session),
            patch.object(agent, "_interruptible_api_call", side_effect=_fake_codex_call),
            patch.object(agent, "_persist_session"),
            patch.object(agent, "_save_trajectory"),
            patch.object(agent, "_cleanup_task_resources"),
            patch(
                "agent.chat_completion_helpers._fallback_entry_unavailable_without_network",
                return_value=None,
            ),
            patch(
                "agent.auxiliary_client.resolve_provider_client",
                return_value=(mock_fb_client, "gpt-5.6-sol"),
            ),
            patch(
                "hermes_cli.model_normalize.normalize_model_for_provider",
                side_effect=lambda m, p: m,
            ),
            patch(
                "agent.model_metadata.get_model_context_length",
                return_value=200000,
            ),
        ):
            result = agent.run_conversation("hello")

        # The SDK was tried exactly once and its wedged session was retired.
        assert session.prompts == ["hello"]
        assert session.closed is True
        assert getattr(agent, "_claude_session", None) is None
        # The chain advanced to the codex entry and it served the turn.
        assert agent.api_mode == "codex_responses"
        assert agent.provider == "openai-codex"
        assert agent.model == "gpt-5.6-sol"
        assert len(codex_calls) == 1
        assert not result.get("failed")
        assert result["completed"] is True
        assert result["final_response"] == "served by codex"
        # Transcript: the trailing user turn was re-served by codex; the
        # limit text never landed as an assistant row in between.
        visible = [m for m in result["messages"] if m.get("role") != "system"]
        assert [m["role"] for m in visible] == ["user", "assistant"]
        assert not any(limit_text in str(m.get("content", "")) for m in visible)

    def test_gate_closed_claude_code_uses_anthropic_wire_with_oauth_detection(self):
        """While the subscription gate is shut, claude-code still means the
        legacy anthropic path — and the OAuth-token detection must apply to
        the aliased slug, not only the literal string "anthropic"."""
        fbs = [{"provider": "claude-code", "model": "claude-sonnet-5"}]
        agent = _make_agent(fallback_model=fbs)
        with (
            patch(
                "hermes_cli.claude_code.subscription_enabled",
                new=lambda config=None: False,
            ),
            patch(
                "agent.chat_completion_helpers._fallback_entry_unavailable_without_network",
                return_value=None,
            ),
            patch(
                "agent.auxiliary_client.resolve_provider_client",
                return_value=(
                    _mock_client(
                        base_url="https://api.anthropic.com",
                        api_key="sk-ant-oat01-test-token",
                    ),
                    "claude-sonnet-5",
                ),
            ),
            patch(
                "hermes_cli.model_normalize.normalize_model_for_provider",
                side_effect=lambda m, p: m,
            ),
            patch(
                "agent.anthropic_adapter.build_anthropic_client",
                return_value=MagicMock(name="anthropic-client"),
            ),
        ):
            assert agent._try_activate_fallback() is True

        assert agent.api_mode == "anthropic_messages"
        assert agent._is_anthropic_oauth is True
