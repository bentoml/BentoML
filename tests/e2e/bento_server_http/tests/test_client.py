import time
import asyncio

import pytest

import bentoml
from bentoml.client import HTTPClient

pytestmark = pytest.mark.asyncio

TIMEOUT = 10


@pytest.fixture(scope="session")
def http_client(simple_service: bentoml.Service, host: str) -> HTTPClient:
    return HTTPClient(simple_service, f"http://{host}")


async def test_async_health(http_client: HTTPClient) -> None:
    start_time = time.monotonic()
    status = ""

    while (time.monotonic() - start_time) < TIMEOUT:
        resp = await http_client.async_health()

        if resp.status == 200:
            status = resp.status
            break

        await asyncio.sleep(3)

    assert status == 200


def test_health(http_client: HTTPClient) -> None:
    start_time = time.monotonic()
    status = ""

    while (time.monotonic() - start_time) < TIMEOUT:
        resp = http_client.health()

        if resp.status == 200:
            status = resp.status
            break

        time.sleep(3)

    assert status == 200
