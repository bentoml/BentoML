import pytest

import bentoml
from bentoml.client import HTTPClient


@pytest.mark.asyncio
async def test_async_health(host: str) -> None:
    http_client = HTTPClient.from_url(f"http://{host}")
    resp = await http_client.async_health()

    assert resp.status == 200


def test_health(host: str) -> None:
    http_client = HTTPClient.from_url(f"http://{host}")
    resp = http_client.health()

    assert resp.status_code == 200


def test_client_request(host: str) -> None:
    http_client = HTTPClient.from_url(f"http://{host}")
    assert http_client.request("GET", "/readyz").status_code == 200


@pytest.mark.asyncio
async def test_async_client(host: str) -> None:
    client = bentoml.client.Client.from_url(f"http://{host}", kind="async-http")

    assert await client.request("GET", "/readyz")

    assert await client.echo_json({"hello": "world"}) == {"hello": "world"}
