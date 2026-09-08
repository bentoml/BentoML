from __future__ import annotations

import time
import typing as t
from pathlib import Path

import pytest

import bentoml

port = 35681
TIMEOUT = 30


def _wait_for_result(task: t.Any) -> str:
    deadline = time.monotonic() + TIMEOUT
    while time.monotonic() < deadline:
        status = task.get_status().value
        if status in ("completed", "failed", "canceled"):
            return status
        time.sleep(0.1)
    raise AssertionError(f"task {task.id} still {task.get_status().value}")


def test_task_submit_status_get_retry(examples: Path) -> None:
    with bentoml.serve(".", working_dir=str(examples / "tasks"), port=port) as server:
        with bentoml.SyncHTTPClient(server.url, server_ready_timeout=100) as client:
            task = client.shout.submit(text="hello")
            assert _wait_for_result(task) == "completed"
            assert task.get() == {"echo": "HELLO"}

            retried = task.retry()
            assert retried.id != task.id
            assert _wait_for_result(retried) == "completed"
            assert retried.get() == {"echo": "HELLO"}


def test_task_submit_with_file(examples: Path, tmp_path: Path) -> None:
    payload = tmp_path / "blob.txt"
    payload.write_text("0123456789")

    with bentoml.serve(
        ".", working_dir=str(examples / "tasks"), port=port + 1
    ) as server:
        with bentoml.SyncHTTPClient(server.url, server_ready_timeout=100) as client:
            task = client.measure.submit(blob=payload)
            assert _wait_for_result(task) == "completed"
            assert task.get() == 10


@pytest.mark.asyncio
async def test_async_task_submit(examples: Path) -> None:
    with bentoml.serve(
        ".", working_dir=str(examples / "tasks"), port=port + 2
    ) as server:
        async with bentoml.AsyncHTTPClient(
            server.url, server_ready_timeout=100
        ) as client:
            task = await client.shout.submit(text="async")
            deadline = time.monotonic() + TIMEOUT
            while time.monotonic() < deadline:
                if (await task.get_status()).value == "completed":
                    break
                time.sleep(0.1)
            assert await task.get() == {"echo": "ASYNC"}
