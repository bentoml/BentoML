import pytest

import bentoml
from bentoml.client import HTTPClient


@pytest.fixture(scope="session")
def http_client(host: str) -> HTTPClient:
    service = bentoml.load(bento_identifier="service:svc")

    return HTTPClient(service, f"http://{host}")


@pytest.mark.asyncio
async def test_async_health(http_client: HTTPClient) -> None:
    resp = await http_client.async_health()

    assert resp.status == 200


def test_health(http_client: HTTPClient) -> None:
    resp = http_client.health()

    assert resp.status == 200
