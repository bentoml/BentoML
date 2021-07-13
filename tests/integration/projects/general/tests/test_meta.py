# pylint: disable=redefined-outer-name
import json

import psutil
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


@pytest.mark.asyncio
async def test_cors(host):
    ORIGIN = "http://bentoml.ai"

    def has_cors_headers(headers):
        assert headers["Access-Control-Allow-Origin"] in ("*", ORIGIN)
        assert "Content-Length" in headers.get("Access-Control-Expose-Headers", [])
        assert "Server" not in headers.get("Access-Control-Expect-Headers", [])
        return True

    await pytest.assert_request(
        "POST",
        f"http://{host}/echo_json",
        headers=(("Content-Type", "application/json"), ("Origin", ORIGIN)),
        data='"hi"',
        assert_status=200,
        assert_headers=has_cors_headers,
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "metrics",
    [
        pytest.param(
            '_mb_request_duration_seconds_count',
            marks=pytest.mark.skipif(
                psutil.MACOS, reason="microbatch metrics is not shown in MacOS tests"
            ),
        ),
        pytest.param(
            '_mb_request_total',
            marks=pytest.mark.skipif(
                psutil.MACOS, reason="microbatch metrics is not shown in MacOS tests"
            ),
        ),
        '_request_duration_seconds_bucket',
    ],
)
async def test_api_server_metrics(host, metrics):
    await pytest.assert_request(
        "POST", f"http://{host}/echo_json", data='"hi"',
    )

    await pytest.assert_request(
        "GET",
        f"http://{host}/metrics",
        assert_status=200,
        assert_data=lambda d: metrics in d.decode(),
    )
