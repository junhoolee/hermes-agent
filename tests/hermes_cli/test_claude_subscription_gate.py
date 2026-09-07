"""Behavior contract for the Claude-subscription release gate.

The gate is imported by the provider catalog and the dashboard web server, so
the invariants that matter are: it is off unless someone explicitly turned it
on, it never raises on a malformed config, and the SDK probe answers even when
the optional `claude-code` extra is not installed.
"""

from hermes_cli import claude_subscription
from hermes_cli.claude_subscription import (
    CLAUDE_AGENT_SDK_MIN_VERSION,
    CLAUDE_CLI_MIN_VERSION,
    DEFAULT_IDLE_SESSION_TTL_SECONDS,
    claude_agent_sdk_available,
    claude_subscription_enabled,
    claude_subscription_idle_session_ttl,
    claude_subscription_start_timeout,
)
from hermes_cli.config_defaults import DEFAULT_CONFIG


def test_shipped_default_is_off():
    """The runtime ships default-off pending Anthropic policy clearance."""
    assert claude_subscription_enabled(DEFAULT_CONFIG) is False


def test_off_for_absent_empty_and_none_config():
    assert claude_subscription_enabled(None) is False
    assert claude_subscription_enabled({}) is False
    assert claude_subscription_enabled({"model": "", "agent": {}}) is False
    assert claude_subscription_enabled({"claude_subscription": {}}) is False


def test_on_only_when_explicitly_enabled():
    assert claude_subscription_enabled({"claude_subscription": {"enabled": True}}) is True
    assert claude_subscription_enabled({"claude_subscription": {"enabled": False}}) is False


def test_malformed_config_reads_as_off_without_raising():
    """A hand-edited config.yaml can put anything under the key; the gate must
    fail closed rather than blow up a startup path."""
    for section in ("yes", 1, [], ["enabled"], None):
        assert claude_subscription_enabled({"claude_subscription": section}) is False
    for config in ("not-a-dict", [], 0):
        assert claude_subscription_enabled(config) is False


def test_availability_probe_returns_bool_and_never_raises():
    assert isinstance(claude_agent_sdk_available(), bool)


def test_availability_probe_is_false_without_the_optional_extra(monkeypatch):
    claude_agent_sdk_available.cache_clear()
    monkeypatch.setattr(claude_subscription.importlib.util, "find_spec", lambda name: None)
    try:
        assert claude_agent_sdk_available() is False
    finally:
        claude_agent_sdk_available.cache_clear()


def test_availability_probe_swallows_a_broken_import_system(monkeypatch):
    """A shadowed/half-installed `claude_agent_sdk` makes find_spec raise. The
    probe runs on startup paths, so it must answer False, not propagate."""
    def _boom(name):
        raise ImportError(name)

    claude_agent_sdk_available.cache_clear()
    monkeypatch.setattr(claude_subscription.importlib.util, "find_spec", _boom)
    try:
        assert claude_agent_sdk_available() is False
    finally:
        claude_agent_sdk_available.cache_clear()


def test_start_timeout_reads_the_configured_seconds():
    assert claude_subscription_start_timeout(
        {"claude_subscription": {"start_timeout": 120}}
    ) == 120.0
    assert claude_subscription_start_timeout(
        {"claude_subscription": {"start_timeout": "90"}}
    ) == 90.0


def test_start_timeout_is_none_when_unset_so_the_runtime_default_applies():
    assert claude_subscription_start_timeout(None) is None
    assert claude_subscription_start_timeout({}) is None
    assert claude_subscription_start_timeout({"claude_subscription": {}}) is None
    assert claude_subscription_start_timeout(DEFAULT_CONFIG) is None


def test_start_timeout_rejects_malformed_values_without_raising():
    """Zero, negative, and non-numeric values read as 'use the default' — a
    hand-edited config.yaml must never wedge session startup."""
    for value in (0, -5, "soon", None, [], {}, False):
        assert claude_subscription_start_timeout(
            {"claude_subscription": {"start_timeout": value}}
        ) is None
    for config in ("not-a-dict", [], 0, {"claude_subscription": "yes"}):
        assert claude_subscription_start_timeout(config) is None


def test_idle_session_ttl_reads_the_configured_seconds():
    assert claude_subscription_idle_session_ttl(
        {"claude_subscription": {"idle_session_ttl_secs": 900}}
    ) == 900.0
    assert claude_subscription_idle_session_ttl(
        {"claude_subscription": {"idle_session_ttl_secs": "60"}}
    ) == 60.0


def test_idle_session_ttl_defaults_when_unset():
    assert claude_subscription_idle_session_ttl(None) == DEFAULT_IDLE_SESSION_TTL_SECONDS
    assert claude_subscription_idle_session_ttl({}) == DEFAULT_IDLE_SESSION_TTL_SECONDS
    assert (
        claude_subscription_idle_session_ttl({"claude_subscription": {}})
        == DEFAULT_IDLE_SESSION_TTL_SECONDS
    )
    assert (
        claude_subscription_idle_session_ttl(DEFAULT_CONFIG)
        == DEFAULT_IDLE_SESSION_TTL_SECONDS
    )


def test_idle_session_ttl_zero_or_negative_disables_reclaim():
    """0 or negative is an explicit opt-out (reclaim disabled), distinct
    from an unset/malformed value which falls back to the default."""
    for value in (0, -5, -0.5):
        assert claude_subscription_idle_session_ttl(
            {"claude_subscription": {"idle_session_ttl_secs": value}}
        ) == 0.0


def test_idle_session_ttl_rejects_malformed_values_without_raising():
    for value in ("soon", None, [], {}, False, True):
        assert claude_subscription_idle_session_ttl(
            {"claude_subscription": {"idle_session_ttl_secs": value}}
        ) == DEFAULT_IDLE_SESSION_TTL_SECONDS
    for config in ("not-a-dict", [], 0, {"claude_subscription": "yes"}):
        assert claude_subscription_idle_session_ttl(config) == DEFAULT_IDLE_SESSION_TTL_SECONDS


def test_pinned_versions_are_orderable_version_strings():
    """Downstream PRs compare an installed version against these floors, so
    both constants must parse as dotted numeric versions."""
    for pin in (CLAUDE_AGENT_SDK_MIN_VERSION, CLAUDE_CLI_MIN_VERSION):
        parts = pin.split(".")
        assert len(parts) >= 3
        assert all(part.isdigit() for part in parts)
