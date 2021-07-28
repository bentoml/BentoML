# pylint: disable=redefined-outer-name
import pytest


@pytest.mark.asyncio
async def test_api_server_meta(host):
    await pytest.assert_request("GET", f"http://{host}/")
    await pytest.assert_request("GET", f"http://{host}/healthz")
    await pytest.assert_request("GET", f"http://{host}/docs.json")


@pytest.mark.asyncio
async def test_no_cors(host):
    ORIGIN = "http://bentoml.ai"

    def no_cors_headers(headers):
        assert headers.get("Access-Control-Allow-Origin") not in ("*", ORIGIN)
        assert "Content-Length" not in headers.get("Access-Control-Expose-Headers", [])
        return True

    await pytest.assert_request(
        "POST",
        f"http://{host}/echo_json",
        headers=(("Content-Type", "application/json"), ("Origin", ORIGIN)),
        data='"hi"',
        assert_status=lambda status: status != 200,
        assert_headers=no_cors_headers,
    )
