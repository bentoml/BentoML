import time
import asyncio

import pytest

import bentoml
from bentoml.client import HTTPClient

pytestmark = pytest.mark.asyncio


@pytest.fixture(scope="session")
def http_client(simple_service: bentoml.Service, host: str) -> HTTPClient:
    return HTTPClient(simple_service, f"http://{host}")


async def test_async_health(http_client: HTTPClient) -> None:
    timeout = 10
    start_time = time.monotonic()
    status = ""

    while (time.monotonic() - start_time) < timeout:
        resp = await http_client.async_health()
        await asyncio.sleep(3)

        if resp.status == 200:
            status = resp.status

    assert status == 200


def test_health(http_client: HTTPClient) -> None:
    timeout = 10
    start_time = time.monotonic()
    status = ""

    while (time.monotonic() - start_time) < timeout:
        resp = http_client.health()
        time.sleep(3)

        if resp.status == 200:
            status = resp.status

    assert status == 200
