import pytest

from bentoml.client import HTTPClient


@pytest.mark.asyncio
async def test_async_health(host: str) -> None:
    http_client = HTTPClient.from_url(f"http://{host}")
    resp = await http_client.async_health()

    assert resp.status == 200


def test_health(host: str) -> None:
    http_client = HTTPClient.from_url(f"http://{host}")
    resp = http_client.health()

    assert resp.status == 200
