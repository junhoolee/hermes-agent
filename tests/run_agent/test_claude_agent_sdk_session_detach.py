"""Unit tests for AIAgent's claude_agent_sdk session detach/release split.

_detach_claude_agent_sdk_session() pops the session reference without doing
any I/O; _release_claude_agent_sdk_session() layers session.close() on top.
The split exists so the gateway's idle sweep (_sweep_idle_cached_agents) can
detach a session out from under a still-cached agent while holding the
agent-cache lock, then run the actual (possibly slow) session.close() on a
separate daemon thread (#93441).
"""

from unittest.mock import MagicMock

from run_agent import AIAgent


def _bare_agent():
    return AIAgent.__new__(AIAgent)


class TestDetachClaudeAgentSdkSession:
    def test_no_session_returns_none(self):
        agent = _bare_agent()
        assert agent._detach_claude_agent_sdk_session() is None

    def test_detach_returns_session_and_clears_it_without_closing(self):
        agent = _bare_agent()
        session = MagicMock()
        agent._claude_session = session

        returned = agent._detach_claude_agent_sdk_session()

        assert returned is session
        assert agent._claude_session is None
        session.close.assert_not_called()

    def test_detach_resets_billing_refusal_sentinel(self):
        from agent.claude_runtime import _UNSET as billing_unset

        agent = _bare_agent()
        agent._claude_session = MagicMock()
        agent._claude_billing_refusal = "refused"

        agent._detach_claude_agent_sdk_session()

        assert agent._claude_billing_refusal is billing_unset

    def test_second_detach_is_a_no_op(self):
        agent = _bare_agent()
        agent._claude_session = MagicMock()

        first = agent._detach_claude_agent_sdk_session()
        second = agent._detach_claude_agent_sdk_session()

        assert first is not None
        assert second is None


class TestReleaseClaudeAgentSdkSessionStillCloses:
    """_release_claude_agent_sdk_session must keep its existing close
    behavior — it's just refactored to go through detach internally."""

    def test_release_still_closes_the_session(self):
        agent = _bare_agent()
        session = MagicMock()
        agent._claude_session = session

        agent._release_claude_agent_sdk_session()

        session.close.assert_called_once()
        assert agent._claude_session is None

    def test_release_with_no_session_does_not_raise(self):
        agent = _bare_agent()
        agent._release_claude_agent_sdk_session()

    def test_release_swallows_close_exception(self):
        agent = _bare_agent()
        session = MagicMock()
        session.close.side_effect = RuntimeError("boom")
        agent._claude_session = session

        agent._release_claude_agent_sdk_session()
