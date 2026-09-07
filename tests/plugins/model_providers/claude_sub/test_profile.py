"""Registration contract for the claude-sub provider profile.

Mirrors ``tests/plugins/model_providers/test_copilot_profile.py``: resolve
the profile through the real discovery path so a regression that swaps the
registered class for a plain ``ProviderProfile`` (or drops the alias) fails
here instead of silently degrading at runtime.
"""

from __future__ import annotations

import sys

import pytest


@pytest.fixture
def claude_sub_profile():
    """Resolve the registered claude-sub profile via full plugin discovery."""
    import model_tools  # noqa: F401 - triggers provider + toolset discovery
    import providers

    profile = providers.get_provider_profile("claude-sub")
    assert profile is not None, "claude-sub provider profile must be registered"
    return profile


class TestProfileRegistration:
    def test_registered_under_canonical_name(self, claude_sub_profile):
        assert claude_sub_profile.name == "claude-sub"

    def test_alias_resolves_to_same_profile(self, claude_sub_profile):
        import providers

        assert providers.get_provider_profile("claude-sdk-sub") is claude_sub_profile

    def test_auth_type_is_external_process(self, claude_sub_profile):
        assert claude_sub_profile.auth_type == "external_process"

    def test_api_mode_is_chat_completions(self, claude_sub_profile):
        assert claude_sub_profile.api_mode == "chat_completions"

    def test_no_env_vars_required(self, claude_sub_profile):
        assert claude_sub_profile.env_vars == ()

    def test_fallback_models_are_curated(self, claude_sub_profile):
        assert claude_sub_profile.fallback_models == (
            "claude-sonnet-5",
            "claude-opus-5",
            "claude-fable-5-1",
        )

    def test_fetch_models_returns_none(self, claude_sub_profile):
        assert claude_sub_profile.fetch_models(api_key=None, base_url=None) is None

    def test_build_extra_body_with_session_id(self, claude_sub_profile):
        assert claude_sub_profile.build_extra_body(session_id="abc123") == {
            "hermes_session_id": "abc123"
        }

    def test_build_extra_body_without_session_id(self, claude_sub_profile):
        assert claude_sub_profile.build_extra_body(session_id=None) == {}

    def test_create_client_returns_claude_sub_client(self, claude_sub_profile):
        from plugins.model_providers.claude_sub.client import ClaudeSubClient

        client = claude_sub_profile.create_client(api_key="claude-sub", base_url="claude-sub://sdk")
        assert isinstance(client, ClaudeSubClient)


class TestProviderRegistryIntegration:
    """The registration seam this profile relies on (hermes_cli/auth.py)."""

    def test_provider_registry_absorbs_external_process_profile(self, claude_sub_profile):
        import model_tools  # noqa: F401
        from hermes_cli.auth import PROVIDER_REGISTRY

        pconfig = PROVIDER_REGISTRY.get("claude-sub")
        assert pconfig is not None
        assert pconfig.auth_type == "external_process"

    def test_provider_registry_absorbs_alias(self, claude_sub_profile):
        import model_tools  # noqa: F401
        from hermes_cli.auth import PROVIDER_REGISTRY

        assert PROVIDER_REGISTRY.get("claude-sdk-sub") is PROVIDER_REGISTRY.get("claude-sub")

    def test_is_external_process_provider(self, claude_sub_profile):
        import model_tools  # noqa: F401
        from hermes_cli.runtime_provider import _is_external_process_provider

        assert _is_external_process_provider("claude-sub") is True


class TestProcessCommandResolution:
    """``_resolve_process_command`` — bundled CLI path, else a bare lookup."""

    def test_falls_back_to_bare_claude_when_sdk_absent(self, monkeypatch, load_plugin_module):
        monkeypatch.setitem(sys.modules, "claude_agent_sdk", None)
        module = load_plugin_module("__init__")
        assert module._resolve_process_command() == "claude"

    def test_prefers_bundled_cli_when_present(self, monkeypatch, tmp_path, fake_sdk, load_plugin_module):
        bundled_dir = tmp_path / "_bundled"
        bundled_dir.mkdir()
        bundled_cli = bundled_dir / "claude"
        bundled_cli.write_text("#!/bin/sh\necho fake\n")

        fake_init = tmp_path / "__init__.py"
        fake_init.write_text("")
        fake_sdk.__file__ = str(fake_init)

        module = load_plugin_module("__init__")
        assert module._resolve_process_command() == str(bundled_cli)

    def test_falls_back_to_bare_claude_when_bundled_missing(
        self, monkeypatch, tmp_path, fake_sdk, load_plugin_module
    ):
        fake_init = tmp_path / "__init__.py"
        fake_init.write_text("")
        fake_sdk.__file__ = str(fake_init)

        module = load_plugin_module("__init__")
        assert module._resolve_process_command() == "claude"
