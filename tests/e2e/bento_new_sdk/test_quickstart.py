import sys
from pathlib import Path

import pytest

import bentoml


def test_serve_and_prediction(examples: Path) -> None:
    server = bentoml.HTTPServer(str(examples / "quickstart"), port=5678)
    server.start(stdout=sys.stdout, stderr=sys.stderr, text=True)

    try:
        with bentoml.SyncHTTPClient("http://127.0.0.1:5678") as client:
            result = client.classify([[4.9, 3.0, 1.4, 0.2]])
        assert result == [0]
    finally:
        server.stop()


@pytest.mark.asyncio
async def test_async_serve_and_prediction(examples: Path) -> None:
    server = bentoml.HTTPServer(str(examples / "quickstart"), port=5678)
    server.start()

    try:
        async with bentoml.AsyncHTTPClient("http://127.0.0.1:5678") as client:
            result = await client.classify([[4.9, 3.0, 1.4, 0.2]])
        assert result == [0]
    finally:
        server.stop()


def test_local_prediction(examples: Path) -> None:
    service = bentoml.load(str(examples / "quickstart"))()

    result = service.classify([[4.9, 3.0, 1.4, 0.2]])
    assert result == [0]


def test_build_and_prediction(examples: Path) -> None:
    bento = bentoml.bentos.build(
        "service.py:IrisClassifier", build_ctx=str(examples / "quickstart")
    )
    server = bentoml.HTTPServer(
        str(bento.tag), port=5678, working_dir=str(examples / "quickstart")
    )
    server.start()

    try:
        with bentoml.SyncHTTPClient("http://127.0.0.1:5678") as client:
            result = client.classify([[4.9, 3.0, 1.4, 0.2]])
        assert result == [0]
    finally:
        server.stop()
