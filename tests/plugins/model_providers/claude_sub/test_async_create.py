"""v0.1-H — ``_ChatCompletions.create`` dual-mode: sync on a plain thread, an
``asyncio.to_thread`` awaitable when the calling thread has a running event
loop.

Regression coverage for the ``TypeError: object types.SimpleNamespace can't
be used in 'await' expression`` bug: ``agent/auxiliary_client.py``'s
``_acreate`` does ``await client.chat.completions.create(**kwargs)`` from
inside a running event loop (the vision auxiliary path). Before v0.1-H,
``create()`` always returned the plain response object, which is not
awaitable. These tests never touch the real SDK — ``_create_chat_completion``
is replaced with a scripted fake, so only the dual-mode dispatch in
``client.py`` is under test.
"""

from __future__ import annotations

import asyncio
import inspect
import threading
import time
from types import SimpleNamespace

import pytest


@pytest.fixture
def client_module(load_plugin_module):
    return load_plugin_module("client")


@pytest.fixture
def client_obj(client_module):
    return client_module.ClaudeSubClient(api_key="claude-sub", base_url="claude-sub://sdk")


def _make_fake(calls, *, sleep_s: float = 0.0):
    def _fake(**kwargs):
        if sleep_s:
            time.sleep(sleep_s)
        calls.append({"kwargs": kwargs, "thread_id": threading.get_ident()})
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content="ok"))],
            model="claude-sonnet-5",
        )

    return _fake


class TestCreateIsAwaitableUnderRunningLoop:
    def test_create_is_awaitable_under_running_loop(self, monkeypatch, client_obj):
        calls: list[dict] = []
        monkeypatch.setattr(client_obj, "_create_chat_completion", _make_fake(calls))

        async def main():
            result = client_obj.chat.completions.create(
                model="m",
                messages=[{"role": "user", "content": "hi"}],
                temperature=0.1,
                timeout=5,
            )
            assert inspect.isawaitable(result)
            return await result

        response = asyncio.run(main())
        assert response.choices[0].message.content == "ok"
        assert len(calls) == 1
        assert calls[0]["kwargs"] == {
            "model": "m",
            "messages": [{"role": "user", "content": "hi"}],
            "temperature": 0.1,
            "timeout": 5,
        }
        assert calls[0]["thread_id"] != threading.get_ident()


class TestCreateIsSyncWithoutLoop:
    def test_create_is_sync_without_loop(self, monkeypatch, client_obj):
        calls: list[dict] = []
        monkeypatch.setattr(client_obj, "_create_chat_completion", _make_fake(calls))

        caller_thread_id = threading.get_ident()
        result = client_obj.chat.completions.create(
            model="m", messages=[{"role": "user", "content": "hi"}]
        )

        assert not inspect.isawaitable(result)
        assert result.choices[0].message.content == "ok"
        assert len(calls) == 1
        assert calls[0]["thread_id"] == caller_thread_id


class TestAwaitDoesNotBlockLoop:
    def test_await_does_not_block_loop(self, monkeypatch, client_obj):
        calls: list[dict] = []
        monkeypatch.setattr(client_obj, "_create_chat_completion", _make_fake(calls, sleep_s=0.3))
        tick_count = {"n": 0}

        async def ticker(done: asyncio.Event):
            while not done.is_set():
                await asyncio.sleep(0.01)
                tick_count["n"] += 1

        async def main():
            done = asyncio.Event()

            async def run_create():
                result = await client_obj.chat.completions.create(
                    model="m", messages=[{"role": "user", "content": "hi"}]
                )
                done.set()
                return result

            create_result, _ = await asyncio.gather(run_create(), ticker(done))
            return create_result

        response = asyncio.run(main())
        assert response.choices[0].message.content == "ok"
        assert tick_count["n"] >= 5


class TestCoreToAsyncClientReturnsSameObjectAndAwaits:
    def test_core_to_async_client_returns_same_object_and_awaits(self, monkeypatch, client_obj):
        from agent.auxiliary_client import _to_async_client

        calls: list[dict] = []
        monkeypatch.setattr(client_obj, "_create_chat_completion", _make_fake(calls))

        aclient, model = _to_async_client(client_obj, "claude-sonnet-5")
        assert aclient is client_obj

        async def main():
            return await aclient.chat.completions.create(
                model=model, messages=[{"role": "user", "content": "hi"}]
            )

        response = asyncio.run(main())
        assert response.choices[0].message.content == "ok"
        assert len(calls) == 1
