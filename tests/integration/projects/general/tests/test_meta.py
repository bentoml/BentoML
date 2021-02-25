# pylint: disable=redefined-outer-name
import json

import pytest


@pytest.mark.asyncio
async def test_api_server_meta(host):
    await pytest.assert_request("GET", f"http://{host}/")
    await pytest.assert_request("GET", f"http://{host}/healthz")
    await pytest.assert_request("GET", f"http://{host}/docs.json")


@pytest.since_bentoml_version("0.11.0+0")
@pytest.mark.asyncio
async def test_customized_route(host):
    CUSTOM_ROUTE = "$~!@%^&*()_-+=[]\\|;:,./predict"

    def path_in_docs(response_body):
        d = json.loads(response_body.decode())
        return f"/{CUSTOM_ROUTE}" in d['paths']

    await pytest.assert_request(
        "GET",
        f"http://{host}/docs.json",
        headers=(("Content-Type", "application/json"),),
        assert_data=path_in_docs,
    )

    await pytest.assert_request(
        "POST",
        f"http://{host}/{CUSTOM_ROUTE}",
        headers=(("Content-Type", "application/json"),),
        data=json.dumps("hello"),
        assert_data=bytes('"hello"', 'ascii'),
    )
