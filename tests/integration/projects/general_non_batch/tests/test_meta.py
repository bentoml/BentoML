# pylint: disable=redefined-outer-name
import pytest


@pytest.mark.asyncio
async def test_api_server_meta(host):
    await pytest.assert_request("GET", f"http://{host}/")
    await pytest.assert_request("GET", f"http://{host}/healthz")
    await pytest.assert_request("GET", f"http://{host}/docs.json")


@pytest.mark.asyncio
async def test_customized_request_schema(host):
    def has_customized_schema(doc_bytes):
        json_str = doc_bytes.decode()
        return "field1" in json_str

    await pytest.assert_request(
        "GET",
        f"http://{host}/docs.json",
        headers=(("Content-Type", "application/json"),),
        assert_data=has_customized_schema,
    )
