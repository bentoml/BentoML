import asyncio
import subprocess
import sys
from pathlib import Path
from time import sleep

import pytest

import bentoml

port = 35678

EXAMPLE_INPUT = "Breaking News: In an astonishing turn of events, the small \
town of Willow Creek has been taken by storm as local resident Jerry Thompson's cat, \
Whiskers, performed what witnesses are calling a 'miraculous and gravity-defying leap.' \
Eyewitnesses report that Whiskers, an otherwise unremarkable tabby cat, jumped \
a record-breaking 20 feet into the air to catch a fly. The event, which took \
place in Thompson's backyard, is now being investigated by scientists for potential \
breaches in the laws of physics. Local authorities are considering a town festival \
to celebrate what is being hailed as 'The Leap of the Century."


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
    )

    await asyncio.sleep(30)

    try:
        with bentoml.SyncHTTPClient(f"http://127.0.0.1:{port}") as client:
            result = client.summarize([EXAMPLE_INPUT])[0]
        assert "Whiskers" in result

        async with bentoml.AsyncHTTPClient(f"http://127.0.0.1:{port}") as client:
            result = (await client.summarize([EXAMPLE_INPUT]))[0]
        assert "Whiskers" in result
    finally:
        server.terminate()


def test_local_prediction(examples: Path) -> None:
    service = bentoml.load(str(examples / "quickstart"))()
    result = service.summarize([EXAMPLE_INPUT])[0]
    assert "Whiskers" in result


def test_build_and_prediction(examples: Path) -> None:
    bento = bentoml.bentos.build(
        "service.py:Summarization", build_ctx=str(examples / "quickstart")
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
    )

    sleep(30)

    try:
        with bentoml.SyncHTTPClient(f"http://127.0.0.1:{port}") as client:
            result = client.summarize([EXAMPLE_INPUT])[0]
        assert "Whiskers" in result
    finally:
        server.terminate()
