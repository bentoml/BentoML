from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from bentoml._internal.server.runner_app import stream_encoder
from bentoml.io import SSE

if TYPE_CHECKING:
    from typing import AsyncGenerator
    from typing import AsyncIterator


async def reverse_proxy(
    iterator: AsyncIterator[bytes],
    chunk_size: int = 5,
) -> AsyncGenerator[bytes, None]:
    """
    A function that will split the data into chunks. This is a mimic of a
    real HTTP reverse proxy that will split the data into chunks.
    """
    async for data in iterator:
        if len(data) > chunk_size:
            while data:
                yield data[:chunk_size]
                data = data[chunk_size:]
        else:
            yield data


async def stream_decoder(
    iterator: AsyncIterator[bytes],
) -> AsyncGenerator[str, None]:
    """
    Converting bytes to string
    """
    async for data in iterator:
        yield data.decode("utf-8")


@pytest.mark.asyncio  # type: ignore
async def test_sse():
    # \n\n is a forbidden sequence in SSE
    test_datas = [
        SSE(data=""),
        SSE(data=" "),
        SSE(data=":"),
        SSE(data="long data i am longer than buffer"),
        SSE(data="my data", event="my event"),
        SSE(data="my data", id="a123cbx"),
        SSE(data="my data", retry=123),
        SSE(data="my data", event="my event", id="a123cbx", retry=123),
        SSE(data="my data\nanother line", event="my event", id="a123cbx", retry=123),
        SSE(
            data="my data\nanother line\nand another line",
            event="my event",
            id="a123cbx",
            retry=123,
        ),
    ]

    async def runner_func():
        for data in test_datas:
            yield data.marshal()

    runner_iterator = stream_decoder(
        reverse_proxy(stream_encoder(runner_func()), chunk_size=5)
    )

    async def api_server_func() -> AsyncGenerator[SSE, None]:
        async for data in SSE.from_iterator(runner_iterator):
            yield data

    results = [data async for data in api_server_func()]
    expected_results = [data for data in test_datas]
    assert results == expected_results
