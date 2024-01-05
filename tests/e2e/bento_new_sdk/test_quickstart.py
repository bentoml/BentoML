import subprocess
import sys
import time
from pathlib import Path

import pytest

import bentoml
from bentoml.client import SyncClient

port = 35678


@pytest.mark.asyncio
async def test_async_serve_and_prediction(examples: Path) -> None:
    server = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "bentoml",
            "serve",
            ".",
            "--working-dir",
            str(examples / "quickstart"),
            "--port",
            str(port),
        ],
        stdout=sys.stdout,
        stderr=sys.stderr,
        text=True,
    )

    try:
        time.sleep(5)
        if server.poll() is not None:
            raise RuntimeError("Server failed to start")
        SyncClient.wait_until_server_ready(
            host="http://127.0.0.1", port=port, timeout=60
        )

        with bentoml.SyncHTTPClient(f"http://127.0.0.1:{port}") as client:
            result = client.classify([[4.9, 3.0, 1.4, 0.2]])
        assert result == [0]

        async with bentoml.AsyncHTTPClient(f"http://127.0.0.1:{port}") as client:
            result = await client.classify([[4.9, 3.0, 1.4, 0.2]])
        assert result == [0]
    finally:
        server.terminate()


def test_local_prediction(examples: Path) -> None:
    service = bentoml.load(str(examples / "quickstart"))()
    result = service.classify([[4.9, 3.0, 1.4, 0.2]])
    assert result == [0]


def test_build_and_prediction(examples: Path) -> None:
    bento = bentoml.bentos.build(
        "service.py:IrisClassifier", build_ctx=str(examples / "quickstart")
    )
    server = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "bentoml",
            "serve",
            str(bento.tag),
            "--port",
            f"{port}",
        ],
        stdout=sys.stdout,
        stderr=sys.stderr,
        text=True,
    )

    try:
        SyncClient.wait_until_server_ready(
            host="http://127.0.0.1", port=port, timeout=60
        )
        with bentoml.SyncHTTPClient(f"http://127.0.0.1:{port}") as client:
            result = client.classify([[4.9, 3.0, 1.4, 0.2]])
        assert result == [0]
    finally:
        server.terminate()
