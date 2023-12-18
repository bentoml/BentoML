from __future__ import annotations

import pytest
from typing import TYPE_CHECKING
from bentoml.io import from_sse, to_sse

if TYPE_CHECKING:
    from typing import AsyncGenerator
    from typing import AsyncIterator


async def reverse_proxy(
    iterator: AsyncIterator[str],
    chunk_size: int = 5,
) -> AsyncGenerator[str, str]:
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


@pytest.mark.asyncio  # type: ignore
async def test_sse():
    test_datas = [
        "",
        " ",
        ":",
        "\n",
        "\n: ",
        "\n\n",
        "\n\n\n\n\n\n\n",
        "1111111111111111111",
        "\n111111111\n222222222",
        "\n\n111111111\n\n222222222",
        "111111111\n\ndata: 222222222",
        '{"data": "111111111"}',
    ]

    async def runner_func():
        for data in test_datas:
            yield to_sse(data)

    runner_iterator = reverse_proxy(runner_func(), chunk_size=5)

    async def api_server_func() -> AsyncGenerator[str, None]:
        async for event in from_sse(runner_iterator):
            yield event.get("data", "")

    results = [data async for data in api_server_func()]
    assert results == test_datas
