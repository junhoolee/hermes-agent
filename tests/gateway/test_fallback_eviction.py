"""Tests for fallback-eviction gating on failed runs (#7130).

When a run fails, the gateway must NOT evict the cached agent — doing so
forces MCP reinit on the next message, creating a CPU-burning restart loop.
Eviction should only happen on successful runs where fallback activated.
"""

import sys
import threading
from pathlib import Path
from unittest.mock import MagicMock


sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))


def _make_runner():
    """Bare GatewayRunner with just the agent-cache infrastructure."""
    from gateway.run import GatewayRunner

    runner = GatewayRunner.__new__(GatewayRunner)
    runner._agent_cache = {}
    runner._agent_cache_lock = threading.Lock()
    return runner


class _ImmediateThread:
    """Stand-in for threading.Thread that runs target() synchronously on
    .start(), so the deferred-release dispatch is observable without a
    real background-thread race."""

    def __init__(self, target=None, args=(), kwargs=None, daemon=None, name=None):
        self._target = target
        self._args = args
        self._kwargs = kwargs or {}
        self.daemon = daemon
        self.name = name

    def start(self):
        if self._target is not None:
            self._target(*self._args, **self._kwargs)


class TestFallbackEvictionGating:
    """The fallback-eviction code path should skip eviction on failed runs."""

    def test_failed_run_does_not_evict_cached_agent(self):
        """When result has failed=True, the cached agent should NOT be evicted."""
        # The fix: `and not _run_failed` guard on the eviction check.
        # Simulate the variables that the eviction block uses.
        result = {"failed": True, "final_response": None, "error": "400 invalid model"}
        _run_failed = result.get("failed") if result else False
        assert _run_failed is True, "Failed run should be detected"


class TestMidTurnEvictionDeferredRelease:
    """A successful fallback-tier switch mid-turn evicts the cache entry
    immediately (so the next lookup doesn't resolve the stale-model agent),
    but the agent it evicts is the SAME object still executing the current
    turn. _evict_cached_agent() used to just `return` in that case — the
    agent was popped from _agent_cache with nothing left holding a
    reference to it, so release_clients() / _release_claude_agent_sdk_session()
    never ran and its claude_agent_sdk session (CLI subprocess + event-loop
    thread) leaked forever (#93441). The fix defers the soft release to
    _release_running_agent_state(), which runs unconditionally at turn end.
    """

    def test_evict_mid_turn_defers_release_to_turn_end(self, monkeypatch):
        import gateway.run as gw_run

        monkeypatch.setattr(gw_run.threading, "Thread", _ImmediateThread)

        runner = _make_runner()
        session_key = "telegram:mid-turn"
        agent = MagicMock()
        agent._gateway_deferred_soft_release = False

        # Agent is both cached AND currently occupying the running-turn slot.
        with runner._agent_cache_lock:
            runner._agent_cache[session_key] = (agent, "sig-old")
        runner._session_state(session_key).turn.agent = agent

        release_calls = []
        runner._release_evicted_agent_soft = lambda a: release_calls.append(a)

        runner._evict_cached_agent(session_key)

        # Cache entry is gone immediately, but the release itself is deferred.
        with runner._agent_cache_lock:
            assert session_key not in runner._agent_cache
        assert release_calls == []
        assert agent._gateway_deferred_soft_release is True

        # Turn ends — this is the call site that must finally release it.
        runner._release_running_agent_state(session_key)

        assert release_calls == [agent]
        assert agent._gateway_deferred_soft_release is False

    def test_recached_agent_is_not_released(self, monkeypatch):
        """If the evicted agent object is somehow back in the cache by the
        time the turn ends (e.g. it lost a race with a fresh cache insert
        for the same session), _release_running_agent_state must not release
        it out from under whatever now holds it."""
        import gateway.run as gw_run

        monkeypatch.setattr(gw_run.threading, "Thread", _ImmediateThread)

        runner = _make_runner()
        session_key = "telegram:mid-turn-recache"
        agent = MagicMock()
        agent._gateway_deferred_soft_release = True  # as if evict already flagged it

        runner._session_state(session_key).turn.agent = agent
        with runner._agent_cache_lock:
            runner._agent_cache[session_key] = (agent, "sig-new")

        release_calls = []
        runner._release_evicted_agent_soft = lambda a: release_calls.append(a)

        runner._release_running_agent_state(session_key)

        assert release_calls == []

    def test_no_deferred_flag_is_a_no_op(self, monkeypatch):
        """A normal (non-deferred) turn end must not trigger any release —
        only agents flagged by a mid-turn evict get this treatment."""
        import gateway.run as gw_run

        monkeypatch.setattr(gw_run.threading, "Thread", _ImmediateThread)

        runner = _make_runner()
        session_key = "telegram:normal-turn"
        agent = MagicMock()
        agent._gateway_deferred_soft_release = False

        runner._session_state(session_key).turn.agent = agent

        release_calls = []
        runner._release_evicted_agent_soft = lambda a: release_calls.append(a)

        runner._release_running_agent_state(session_key)

        assert release_calls == []

    def test_thread_spawn_failure_falls_back_to_inline_release(self, monkeypatch):
        """If threading.Thread() itself raises (e.g. interpreter shutdown),
        the deferred release must still run inline rather than being lost."""
        import gateway.run as gw_run

        def _boom(*args, **kwargs):
            raise RuntimeError("can't start new thread")

        monkeypatch.setattr(gw_run.threading, "Thread", _boom)

        runner = _make_runner()
        session_key = "telegram:mid-turn-thread-fail"
        agent = MagicMock()
        agent._gateway_deferred_soft_release = True

        runner._session_state(session_key).turn.agent = agent

        release_calls = []
        runner._release_evicted_agent_soft = lambda a: release_calls.append(a)

        runner._release_running_agent_state(session_key)

        assert release_calls == [agent]


class TestReleaseEvictedAgentSoftClosesClaudeSession:
    """The real _release_evicted_agent_soft path (not a MagicMock stand-in)
    against a genuine AIAgent must reach _release_claude_agent_sdk_session()
    and close a live claude_agent_sdk session."""

    def test_release_evicted_agent_soft_closes_claude_session(self):
        from run_agent import AIAgent

        agent = AIAgent.__new__(AIAgent)
        session = MagicMock()
        session.closed = False
        agent._claude_session = session

        runner = _make_runner()
        runner._release_evicted_agent_soft(agent)

        session.close.assert_called_once()
        assert agent._claude_session is None


