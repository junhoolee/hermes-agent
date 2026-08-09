"""Background task credential gate vs keyless runtimes.

The Claude subscription runtime carries ``api_key=""`` by contract
(hermes_cli/runtime_provider.py: the Agent SDK owns the login), so the
background-task refusal at the top of ``_run_background_task_inner`` must not
mistake it for "no provider credentials configured" — the same bug class as
the /compress and session-hygiene gates.
"""

import importlib
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from gateway.config import GatewayConfig, Platform, PlatformConfig
from gateway.platforms.base import BasePlatformAdapter, SendResult
from gateway.session import SessionSource


class _CaptureAdapter(BasePlatformAdapter):
    def __init__(self):
        super().__init__(
            PlatformConfig(enabled=True, token="fake-token"), Platform.TELEGRAM
        )
        self.sent = []

    async def connect(self, *, is_reconnect: bool = False) -> bool:
        return True

    async def disconnect(self) -> None:
        return None

    async def send(self, chat_id, content, reply_to=None, metadata=None) -> SendResult:
        self.sent.append({"chat_id": chat_id, "content": content})
        return SendResult(success=True, message_id="bg-1")

    async def get_chat_info(self, chat_id: str):
        return {"id": chat_id}


class _SentinelStop(Exception):
    """Raised right after the credential gate to end the test run early."""


async def _run_background_task(monkeypatch, runtime_kwargs):
    gateway_run = importlib.import_module("gateway.run")
    runner = object.__new__(gateway_run.GatewayRunner)
    adapter = _CaptureAdapter()
    runner.config = GatewayConfig(
        platforms={Platform.TELEGRAM: PlatformConfig(enabled=True, token="fake-token")}
    )
    runner.adapters = {Platform.TELEGRAM: adapter}
    runner.hooks = SimpleNamespace(emit=AsyncMock(), loaded_hooks=False)

    monkeypatch.setattr(gateway_run, "_load_gateway_config", lambda: {})
    monkeypatch.setattr(
        gateway_run, "_resolve_runtime_agent_kwargs", lambda: dict(runtime_kwargs)
    )

    def _stop_after_gate(*_args, **_kwargs):
        raise _SentinelStop("gate passed")

    # _platform_config_key is the first call after the credential gate — a
    # sentinel there proves the gate was passed without building an agent.
    monkeypatch.setattr(gateway_run, "_platform_config_key", _stop_after_gate)

    source = SessionSource(
        platform=Platform.TELEGRAM, chat_id="c1", chat_type="dm", user_id="u1"
    )
    await runner._run_background_task_inner("do the thing", source, "task-1")
    return adapter.sent


@pytest.mark.asyncio
async def test_background_task_allows_keyless_claude_agent_sdk_runtime(monkeypatch):
    sent = await _run_background_task(
        monkeypatch,
        {
            "api_key": "",
            "api_mode": "claude_agent_sdk",
            "provider": "claude-code",
            "base_url": "claude-sdk://subscription",
        },
    )
    refusals = [m for m in sent if "no provider credentials" in m["content"]]
    assert not refusals, (
        f"keyless claude_agent_sdk runtime refused as credential-less: {refusals}"
    )
    # The sentinel failure proves execution moved PAST the gate.
    assert any("gate passed" in m["content"] for m in sent)


@pytest.mark.asyncio
async def test_background_task_still_refuses_truly_unconfigured_runtime(monkeypatch):
    sent = await _run_background_task(
        monkeypatch,
        {"api_key": "", "api_mode": "chat_completions", "provider": ""},
    )
    assert any("no provider credentials" in m["content"] for m in sent)
