import pytest

import bentoml


@pytest.mark.asyncio
async def test_async_health(host: str) -> None:
    client = bentoml.client.HTTPClient.from_url(f"http://{host}")
    resp = await client.async_health()

    assert resp.status == 200


def test_health(host: str) -> None:
    client = bentoml.client.HTTPClient.from_url(f"http://{host}")
    resp = client.health()

    assert resp.status == 200
