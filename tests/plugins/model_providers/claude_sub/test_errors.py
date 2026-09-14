"""SDK failure -> ``openai`` exception mapping for the claude-sub provider.

Hermes' fallback chain and rate-limit handling key off ``openai`` exception
types and ``.status_code`` (see ``agent/error_classifier.py``), not a
provider-specific exception hierarchy. These tests pin that the synthesized
exceptions carry the right type and status for the classifier to work
end-to-end, without going near a real SDK subprocess.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import openai
import pytest


@dataclass
class _FakeResultMessage:
    is_error: bool = False
    result: str | None = None
    errors: list | None = None
    api_error_status: int | None = None


class TestClassifyResult:
    def test_not_an_error_returns_none(self, load_plugin_module):
        errors = load_plugin_module("errors")
        message = _FakeResultMessage(is_error=False)
        assert errors.classify_result(message) is None

    @pytest.mark.parametrize(
        "text",
        [
            "You have hit your rate limit",
            "rate_limit_exceeded",
            "usage limit reached for this session",
            "quota exceeded",
            "the model is overloaded, please retry",
            "HTTP 529 received",
            "too many requests, slow down",
        ],
    )
    def test_rate_limit_markers_classify_as_429(self, load_plugin_module, text):
        errors = load_plugin_module("errors")
        message = _FakeResultMessage(is_error=True, result=text)
        assert errors.classify_result(message) == 429

    def test_api_error_status_429_wins_even_without_marker(self, load_plugin_module):
        errors = load_plugin_module("errors")
        message = _FakeResultMessage(is_error=True, result="something went wrong", api_error_status=429)
        assert errors.classify_result(message) == 429

    def test_generic_error_classifies_as_503(self, load_plugin_module):
        errors = load_plugin_module("errors")
        message = _FakeResultMessage(is_error=True, result="the CLI crashed unexpectedly")
        assert errors.classify_result(message) == 503

    def test_errors_list_is_considered(self, load_plugin_module):
        errors = load_plugin_module("errors")
        message = _FakeResultMessage(is_error=True, result=None, errors=["quota exceeded for org"])
        assert errors.classify_result(message) == 429


class TestRaiseStatus:
    def test_429_raises_rate_limit_error(self, load_plugin_module):
        errors = load_plugin_module("errors")
        with pytest.raises(openai.RateLimitError) as excinfo:
            errors.raise_status(429, "claude-sub: rate-limit: too many requests")
        assert excinfo.value.status_code == 429

    def test_503_raises_api_status_error(self, load_plugin_module):
        errors = load_plugin_module("errors")
        with pytest.raises(openai.APIStatusError) as excinfo:
            errors.raise_status(503, "claude-sub: error: the CLI crashed")
        assert excinfo.value.status_code == 503

    def test_504_raises_api_status_error(self, load_plugin_module):
        errors = load_plugin_module("errors")
        with pytest.raises(openai.APIStatusError) as excinfo:
            errors.raise_status(504, "claude-sub: timeout: turn exceeded deadline")
        assert excinfo.value.status_code == 504

    def test_classifier_recognizes_synthesized_rate_limit_error(self, load_plugin_module):
        """Integration with Hermes' own classifier — the whole point of the mapping."""
        errors = load_plugin_module("errors")
        from agent.error_classifier import classify_api_error

        try:
            errors.raise_status(429, "claude-sub: rate-limit: too many requests")
        except openai.RateLimitError as exc:
            classified = classify_api_error(exc, provider="claude-sub", model="claude-sonnet-5")
        assert classified.status_code == 429


class TestErrorMessageFor:
    def test_includes_reason_and_text(self, load_plugin_module):
        errors = load_plugin_module("errors")
        message = errors.error_message_for("timeout", "turn exceeded 1800s")
        assert message == "claude-sub: timeout: turn exceeded 1800s"

    def test_without_text(self, load_plugin_module):
        errors = load_plugin_module("errors")
        assert errors.error_message_for("timeout") == "claude-sub: timeout"

    def test_caps_at_500_chars(self, load_plugin_module):
        errors = load_plugin_module("errors")
        long_text = "x" * 1000
        message = errors.error_message_for("error", long_text)
        # "claude-sub: error: " prefix + at most 500 chars of the SDK text.
        assert message == "claude-sub: error: " + ("x" * 500)
