"""Tests for claude_agent_sdk idle-session reclaim in the gateway's cache sweep.

A cached AIAgent survives up to the agent-cache idle TTL (hours, for a
`session_reset: none` conversation) — but its claude_agent_sdk session
(claude CLI child process + event-loop thread) shouldn't have to wait that
long. _sweep_idle_cached_agents() independently detaches and closes a
claude_agent_sdk session once claude_subscription.idle_session_ttl_secs has
elapsed, even while the surrounding agent stays cached (#93441).
"""

import threading
import time
from unittest.mock import MagicMock

import pytest


def _bounded_runner():
    from collections import OrderedDict
    from gateway.run import GatewayRunner

    runner = GatewayRunner.__new__(GatewayRunner)
    runner._agent_cache = OrderedDict()
    runner._agent_cache_lock = threading.Lock()
    return runner


class _ImmediateThread:
    """threading.Thread stand-in that runs target() synchronously on .start()."""

    def __init__(self, target=None, args=(), kwargs=None, daemon=None, name=None):
        self._target = target
        self._args = args
        self._kwargs = kwargs or {}
        self.daemon = daemon
        self.name = name

    def start(self):
        if self._target is not None:
            self._target(*self._args, **self._kwargs)


def _fake_agent(idle_seconds: float, claude_session=None):
    m = MagicMock()
    m._last_activity_ts = time.time() - idle_seconds
    m._claude_session = claude_session
    if claude_session is not None:
        m._detach_claude_agent_sdk_session.return_value = claude_session
    return m


@pytest.fixture(autouse=True)
def _stub_gateway_config(monkeypatch):
    """Isolate the sweep from real config-file I/O; each test supplies its
    own claude_subscription_idle_session_ttl stub."""
    import gateway.run as gw_run

    monkeypatch.setattr(gw_run, "_load_gateway_config", lambda *a, **k: {})
    monkeypatch.setattr(gw_run.threading, "Thread", _ImmediateThread)


def _set_claude_ttl(monkeypatch, value):
    from hermes_cli import claude_subscription as csub

    monkeypatch.setattr(csub, "claude_subscription_idle_session_ttl", lambda cfg: value)


class TestClaudeSdkIdleReclaim:
    def test_claude_session_reclaimed_while_agent_stays_cached(self, monkeypatch):
        """(a) idle exceeds the claude TTL but not the agent's own cache
        idle TTL: the session is detached and closed, but the agent entry
        itself is NOT evicted from the cache."""
        import gateway.run as gw_run

        monkeypatch.setattr(gw_run, "_AGENT_CACHE_IDLE_TTL_SECS", 3600.0)
        _set_claude_ttl(monkeypatch, 30.0)

        runner = _bounded_runner()
        session = MagicMock()
        agent = _fake_agent(idle_seconds=60.0, claude_session=session)
        runner._agent_cache["s1"] = (agent, "sig")

        evicted = runner._sweep_idle_cached_agents()

        assert evicted == 0
        assert "s1" in runner._agent_cache
        agent._detach_claude_agent_sdk_session.assert_called_once()
        session.close.assert_called_once()

    def test_running_agent_is_never_touched(self, monkeypatch):
        """(b) An agent occupying the running-turn slot must be skipped
        entirely — not evicted, and its claude session not even inspected."""
        import gateway.run as gw_run

        monkeypatch.setattr(gw_run, "_AGENT_CACHE_IDLE_TTL_SECS", 3600.0)
        _set_claude_ttl(monkeypatch, 1.0)

        runner = _bounded_runner()
        session = MagicMock()
        agent = _fake_agent(idle_seconds=1000.0, claude_session=session)
        runner._agent_cache["s1"] = (agent, "sig")
        runner._session_state("s1").turn.agent = agent  # mid-turn

        evicted = runner._sweep_idle_cached_agents()

        assert evicted == 0
        assert "s1" in runner._agent_cache
        agent._detach_claude_agent_sdk_session.assert_not_called()
        session.close.assert_not_called()

    def test_zero_ttl_disables_reclaim(self, monkeypatch):
        """(c) claude_subscription.idle_session_ttl_secs <= 0 means: never
        reclaim independently of the agent's own cache TTL."""
        import gateway.run as gw_run

        monkeypatch.setattr(gw_run, "_AGENT_CACHE_IDLE_TTL_SECS", 3600.0)
        _set_claude_ttl(monkeypatch, 0.0)

        runner = _bounded_runner()
        session = MagicMock()
        agent = _fake_agent(idle_seconds=60.0, claude_session=session)
        runner._agent_cache["s1"] = (agent, "sig")

        evicted = runner._sweep_idle_cached_agents()

        assert evicted == 0
        assert "s1" in runner._agent_cache
        agent._detach_claude_agent_sdk_session.assert_not_called()
        session.close.assert_not_called()

    def test_agent_without_claude_session_is_unaffected(self, monkeypatch):
        """An agent that never started a claude_agent_sdk session (e.g. a
        non-claude provider) must not trip the detach path at all."""
        import gateway.run as gw_run

        monkeypatch.setattr(gw_run, "_AGENT_CACHE_IDLE_TTL_SECS", 3600.0)
        _set_claude_ttl(monkeypatch, 1.0)

        runner = _bounded_runner()
        agent = _fake_agent(idle_seconds=60.0, claude_session=None)
        runner._agent_cache["s1"] = (agent, "sig")

        evicted = runner._sweep_idle_cached_agents()

        assert evicted == 0
        assert "s1" in runner._agent_cache
        agent._detach_claude_agent_sdk_session.assert_not_called()

    def test_full_agent_eviction_takes_priority_over_claude_reclaim(self, monkeypatch):
        """When the agent itself is being fully evicted (idle past its own
        cache TTL, session unfinalized/mode-none), the claude-session detach
        must not also run — release_clients() already tears it down."""
        import gateway.run as gw_run

        monkeypatch.setattr(gw_run, "_AGENT_CACHE_IDLE_TTL_SECS", 5.0)
        _set_claude_ttl(monkeypatch, 1.0)

        runner = _bounded_runner()
        runner.session_store = MagicMock()
        runner.session_store._entries = {}

        session = MagicMock()
        agent = _fake_agent(idle_seconds=60.0, claude_session=session)
        release_calls = []
        runner._release_evicted_agent_soft = lambda a: release_calls.append(a)
        runner._agent_cache["s1"] = (agent, "sig")

        evicted = runner._sweep_idle_cached_agents()

        assert evicted == 1
        assert "s1" not in runner._agent_cache
        assert release_calls == [agent]
        agent._detach_claude_agent_sdk_session.assert_not_called()
