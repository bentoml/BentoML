from typing import Optional

import pytest

import bentoml
from _bentoml_impl.client.proxy2 import AsyncClient


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "args, expected", [((), b"7"), ((None,), b"null"), ((0,), b"0")]
)
async def test_typed_client_root_input_default(args, expected):
    @bentoml.service
    class TestService:
        @bentoml.api
        def echo(self, value: Optional[int] = 7, /) -> Optional[int]:
            return value

    client = AsyncClient("http://test", service=TestService)
    try:
        payload = client._build_payload(client.endpoints["echo"], args, {}, {})
        assert b"".join([chunk async for chunk in payload]) == expected
    finally:
        await client.close()
