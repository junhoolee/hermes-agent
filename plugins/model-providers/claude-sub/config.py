"""Settings for the claude-sub provider plugin.

Reads an optional top-level ``claude_sub`` dict from config.yaml (never added
to Hermes' own ``DEFAULT_CONFIG`` — this plugin owns its own defaults). Any
missing key, wrong type, or read failure falls back to the built-in default;
a broken/absent config must never prevent the plugin from working.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Settings:
    start_timeout: float = 60.0
    turn_timeout: float = 1800.0
    stall_timeout: float | None = 300.0
    orphan_timeout: float = 120.0
    bootstrap_max_chars: int = 60000
    identity_append: str = ""


def _coerce_float(value, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _coerce_optional_float(value, default: float | None) -> float | None:
    if value is None:
        return default
    try:
        coerced = float(value)
    except (TypeError, ValueError):
        return default
    return None if coerced == 0 else coerced


def _coerce_int(value, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _coerce_str(value, default: str) -> str:
    return value if isinstance(value, str) else default


def load_settings() -> Settings:
    """Load claude_sub settings from config.yaml, falling back to defaults."""
    raw: dict = {}
    try:
        from hermes_cli.config import load_config_readonly

        cfg = load_config_readonly()
        candidate = cfg.get("claude_sub") if isinstance(cfg, dict) else None
        if isinstance(candidate, dict):
            raw = candidate
    except Exception:
        raw = {}

    defaults = Settings()
    return Settings(
        start_timeout=_coerce_float(raw.get("start_timeout"), defaults.start_timeout),
        turn_timeout=_coerce_float(raw.get("turn_timeout"), defaults.turn_timeout),
        stall_timeout=_coerce_optional_float(raw.get("stall_timeout"), defaults.stall_timeout),
        orphan_timeout=_coerce_float(raw.get("orphan_timeout"), defaults.orphan_timeout),
        bootstrap_max_chars=_coerce_int(raw.get("bootstrap_max_chars"), defaults.bootstrap_max_chars),
        identity_append=_coerce_str(raw.get("identity_append"), defaults.identity_append),
    )
