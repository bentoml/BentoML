import pytest

from bentoml.client import AsyncHTTPClient
from bentoml.client import HTTPClient


@pytest.mark.asyncio
async def test_async_health(host: str) -> None:
    http_client = await AsyncHTTPClient.from_url(f"http://{host}")
    resp = await http_client.health()

    assert resp.status_code == 200


def test_health(host: str) -> None:
    http_client = HTTPClient.from_url(f"http://{host}")
    resp = http_client.health()

    assert resp.status_code == 200


def test_text_endpoint(host: str) -> None:
    http_client = HTTPClient.from_url(f"http://{host}")
    resp = http_client.yo("Bob")

    assert resp == "yo Bob"
