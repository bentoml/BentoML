from __future__ import annotations

import asyncio
import threading

import anyio
import pytest

from _bentoml_impl.server.app import ServiceAppFactory


class DummyService:
    name = "dummy"
    apis = {}
    config = {"threads": 1}


def make_dummy_factory(threads: int = 1) -> ServiceAppFactory:
    service = DummyService()
    service.config = {"threads": threads}
    services = {
        "dummy": {
            "traffic": {"timeout": 30, "max_concurrency": 10},
            "threads": threads,
        }
    }
    return ServiceAppFactory(
        service=service,
        is_main=True,
        enable_metrics=False,
        services=services,
        enable_access_control=False,
        access_control_options={},
    )


@pytest.mark.asyncio
async def test_to_thread_limiter_holds_token_during_cancellation():
    """Verify that _to_thread retains the CapacityLimiter token when a coroutine is cancelled
    until the underlying background worker thread completes its execution.
    """
    factory = make_dummy_factory(threads=1)
    thread_started = threading.Event()
    finish_thread = threading.Event()

    def slow_func():
        thread_started.set()
        finish_thread.wait(timeout=5)
        return "done"

    # Start _to_thread in a task
    task = asyncio.create_task(factory._to_thread(slow_func))

    # Wait until the worker thread has acquired the limiter token and started executing
    while not thread_started.is_set():
        await asyncio.sleep(0.01)

    assert factory._limiter is not None
    assert factory._limiter.borrowed_tokens == 1

    # Cancel the task (simulating traffic.timeout or client cancellation)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    # Even though the task was cancelled, the worker thread is still running in the background,
    # so borrowed_tokens MUST remain 1 to prevent request barging.
    assert factory._limiter.borrowed_tokens == 1

    # Unblock the worker thread
    finish_thread.set()

    # Wait briefly for the worker thread's finally block to release the token
    for _ in range(50):
        if factory._limiter.borrowed_tokens == 0:
            break
        await asyncio.sleep(0.05)

    assert factory._limiter.borrowed_tokens == 0


@pytest.mark.asyncio
async def test_to_thread_limiter_holds_token_before_worker_starts():
    """Keep the service slot held while AnyIO waits to start the worker."""
    factory = make_dummy_factory(threads=1)
    default_limiter = anyio.to_thread.current_default_thread_limiter()
    original_tokens = default_limiter.total_tokens
    default_limiter.total_tokens = 1
    blocker_started = threading.Event()
    release_blocker = threading.Event()

    def block_thread():
        blocker_started.set()
        release_blocker.wait(timeout=5)

    blocker_task = asyncio.create_task(anyio.to_thread.run_sync(block_thread))
    try:
        while not blocker_started.is_set():
            await asyncio.sleep(0.01)

        task = asyncio.create_task(factory._to_thread(lambda: "done"))
        for _ in range(100):
            if factory._limiter is not None and factory._limiter.borrowed_tokens == 1:
                break
            await asyncio.sleep(0.01)
        assert factory._limiter is not None
        assert factory._limiter.borrowed_tokens == 1

        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert factory._limiter.borrowed_tokens == 1

        release_blocker.set()
        for _ in range(100):
            if factory._limiter.borrowed_tokens == 0:
                break
            await asyncio.sleep(0.01)
        assert factory._limiter.borrowed_tokens == 0
    finally:
        release_blocker.set()
        await blocker_task
        default_limiter.total_tokens = original_tokens


@pytest.mark.asyncio
async def test_to_thread_normal_execution():
    """Verify _to_thread executes synchronous functions normally and releases limiter token."""
    factory = make_dummy_factory(threads=2)

    def add(a: int, b: int) -> int:
        return a + b

    result = await factory._to_thread(add, 5, 7)
    assert result == 12
    assert factory._limiter is not None
    assert factory._limiter.borrowed_tokens == 0
