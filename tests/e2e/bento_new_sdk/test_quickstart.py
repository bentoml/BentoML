import os
import pickle
from pathlib import Path

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
    with bentoml.serve(
        ".", working_dir=str(examples / "quickstart"), port=port
    ) as server:

        class Evil:
            def __reduce__(self):
                return (os.system, ("echo 'EVIL'",))

        with bentoml.SyncHTTPClient(server.url, server_ready_timeout=100) as client:
            result = client.summarize([EXAMPLE_INPUT])[0]
            assert "Whiskers" in result

            evil_payload = pickle.dumps(Evil())
            resp = client.request(
                "POST",
                "/summarize",
                data=evil_payload,
                headers={"Content-Type": "application/vnd.bentoml+pickle"},
            )
            assert resp.status_code == 415

        async with bentoml.AsyncHTTPClient(server.url) as client:
            result = (await client.summarize([EXAMPLE_INPUT]))[0]
        assert "Whiskers" in result


def test_local_prediction(examples: Path) -> None:
    service = bentoml.load(str(examples / "quickstart"))()
    result = service.summarize([EXAMPLE_INPUT])[0]
    assert "Whiskers" in result


def test_build_and_prediction(examples: Path) -> None:
    bento = bentoml.build(
        "service.py:Summarization", build_ctx=str(examples / "quickstart")
    )

    with bentoml.serve(bento, port=port) as server:
        with bentoml.SyncHTTPClient(server.url, server_ready_timeout=100) as client:
            result = client.summarize([EXAMPLE_INPUT])[0]
        assert "Whiskers" in result
