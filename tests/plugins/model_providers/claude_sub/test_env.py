"""Sanitized child environment contract for the claude-sub subprocess.

``ClaudeAgentOptions.env`` can only add/override keys, never delete one — so
the removal of higher-precedence credentials has to happen in the dict this
plugin hands the transport. These tests pin exactly which variables are
stripped, which always survive, and that a caller-supplied ``options.env``
cannot smuggle a blocked credential back in.
"""

from __future__ import annotations

from types import SimpleNamespace


RAW_ENV = {
    "HERMES_KANBAN_TASK": "t_x",
    "HERMES_KANBAN_WORKSPACE": "/w",
    "HERMES_SESSION_ID": "s",
    "HERMES_SESSION_SOURCE": "kanban",
    "HERMES_SINGLE_QUERY_SESSION": "1",
    "ANTHROPIC_API_KEY": "k",
    "CLAUDE_CODE_USE_BEDROCK": "1",
    "HERMES_HOME": "/h",
    "HOME": "/u",
    "PATH": "/p",
}

BLOCKED_KEYS = (
    "HERMES_KANBAN_TASK",
    "HERMES_KANBAN_WORKSPACE",
    "HERMES_SESSION_ID",
    "HERMES_SESSION_SOURCE",
    "HERMES_SINGLE_QUERY_SESSION",
    "ANTHROPIC_API_KEY",
    "CLAUDE_CODE_USE_BEDROCK",
)

SURVIVING_KEYS = {"HERMES_HOME": "/h", "HOME": "/u", "PATH": "/p"}


class TestScrubEnv:
    def test_blocked_keys_removed(self, load_plugin_module):
        env = load_plugin_module("env")
        scrubbed = env.scrub_env(RAW_ENV)
        for key in BLOCKED_KEYS:
            assert key not in scrubbed

    def test_pass_through_keys_survive(self, load_plugin_module):
        env = load_plugin_module("env")
        scrubbed = env.scrub_env(RAW_ENV)
        for key, value in SURVIVING_KEYS.items():
            assert scrubbed[key] == value

    def test_does_not_mutate_input(self, load_plugin_module):
        env = load_plugin_module("env")
        original = dict(RAW_ENV)
        env.scrub_env(RAW_ENV)
        assert RAW_ENV == original


class TestBuildChildEnv:
    def test_blocked_keys_removed_from_child_env(self, load_plugin_module, monkeypatch):
        env = load_plugin_module("env")
        monkeypatch.setattr(env.os, "environ", dict(RAW_ENV))
        options = SimpleNamespace(cwd=None, env={})
        child_env = env.build_child_env(options)
        for key in BLOCKED_KEYS:
            assert key not in child_env

    def test_options_env_cannot_reintroduce_blocked_credential(self, load_plugin_module, monkeypatch):
        env = load_plugin_module("env")
        monkeypatch.setattr(env.os, "environ", dict(RAW_ENV))
        options = SimpleNamespace(cwd=None, env={"ANTHROPIC_API_KEY": "x"})
        child_env = env.build_child_env(options)
        assert "ANTHROPIC_API_KEY" not in child_env

    def test_mcp_tool_timeout_default_set(self, load_plugin_module, monkeypatch):
        env = load_plugin_module("env")
        monkeypatch.setattr(env.os, "environ", dict(RAW_ENV))
        options = SimpleNamespace(cwd=None, env={})
        child_env = env.build_child_env(options)
        assert child_env["MCP_TOOL_TIMEOUT"] == "3600000"

    def test_mcp_tool_timeout_respects_existing_value(self, load_plugin_module, monkeypatch):
        env = load_plugin_module("env")
        raw = dict(RAW_ENV)
        raw["MCP_TOOL_TIMEOUT"] = "60000"
        monkeypatch.setattr(env.os, "environ", raw)
        options = SimpleNamespace(cwd=None, env={})
        child_env = env.build_child_env(options)
        assert child_env["MCP_TOOL_TIMEOUT"] == "60000"

    def test_home_unchanged(self, load_plugin_module, monkeypatch):
        env = load_plugin_module("env")
        monkeypatch.setattr(env.os, "environ", dict(RAW_ENV))
        options = SimpleNamespace(cwd=None, env={})
        child_env = env.build_child_env(options)
        assert child_env["HOME"] == "/u"

    def test_cwd_sets_pwd(self, load_plugin_module, monkeypatch):
        env = load_plugin_module("env")
        monkeypatch.setattr(env.os, "environ", dict(RAW_ENV))
        options = SimpleNamespace(cwd="/work/dir", env={})
        child_env = env.build_child_env(options)
        assert child_env["PWD"] == "/work/dir"

    def test_entrypoint_marker_set(self, load_plugin_module, monkeypatch):
        env = load_plugin_module("env")
        monkeypatch.setattr(env.os, "environ", dict(RAW_ENV))
        options = SimpleNamespace(cwd=None, env={})
        child_env = env.build_child_env(options)
        assert child_env["CLAUDE_CODE_ENTRYPOINT"] == "sdk-py"
