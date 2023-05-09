from __future__ import annotations

import time
import typing as t
import asyncio
from pathlib import Path

import pytest
from starlette.datastructures import Headers

import bentoml

if t.TYPE_CHECKING:
    F = t.Callable[..., t.Coroutine[t.Any, t.Any, t.Any]]


@pytest.mark.asyncio
async def test_api_server_load(arequest: F):
    for _ in range(20):
        await arequest(
            api_name="echo_json",
            headers={"Content-Type": "application/json"},
            data='"hi"',
            assert_output='"hi"',
        )


@pytest.mark.asyncio
async def test_api_server_meta(host: str) -> None:
    client = bentoml.client.HTTPClient.from_url(f"http://{host}")
    resp = await client.async_request("GET", "/")
    assert resp.ok and resp.status == 200
    resp = await client.async_request("GET", "/healthz")
    assert resp.ok and resp.status == 200
    resp = await client.async_request("GET", "/livez")
    assert resp.ok and resp.status == 200
    resp = await client.async_request("GET", "/ping")
    assert resp.ok and resp.status == 200
    resp = await client.async_request("GET", "/hello")
    assert resp.status == 200 and await resp.read() == b'{"Hello":"World"}'
    resp = await client.async_request("GET", "/docs.json")
    assert resp.ok and resp.status == 200
    resp = await client.async_request("GET", "/metrics")
    assert resp.ok and resp.status == 200 and await resp.read()
    resp = await client.async_request("POST", "/api/v1/with_prefix")
    assert resp.status == 404


@pytest.mark.asyncio
async def test_context(host: str):
    client = bentoml.client.HTTPClient.from_url(f"http://{host}")
    resp = await client.async_request("POST", "/use_context?error=yes")
    assert resp.status == 400
    assert await resp.read() == b"yes"


@pytest.mark.asyncio
async def test_runner_readiness(host: str) -> None:
    client = bentoml.client.HTTPClient.from_url(f"http://{host}")
    timeout = 20
    start_time = time.time()
    while (time.time() - start_time) < timeout:
        resp = await client.async_request("GET", "/readyz")
        await asyncio.sleep(5)
        if resp.status == 200:
            break
    assert resp.status == 200


@pytest.mark.asyncio
async def test_cors(host: str, server_config_file: str) -> None:
    client = bentoml.client.HTTPClient.from_url(f"http://{host}")
    ORIGIN = "http://bentoml.ai:8080"

    resp = await client.async_request(
        "OPTIONS",
        "/echo_json",
        headers={
            "Content-Type": "application/json",
            "Origin": ORIGIN,
            "Access-Control-Request-Method": "POST",
            "Access-Control-Request-Headers": "Content-Type",
        },
    )

    # all test configs lives under ../configs, but we are only interested in cors.
    fname = Path(server_config_file).name

    if fname == "cors_enabled.yml":
        assert resp.status == 200
    else:
        assert resp.status != 200

    resp = await client.async_request(
        "POST",
        "/echo_json",
        headers={"Content-Type": "application/json", "Origin": ORIGIN},
        data='"hi"',
    )
    if fname == "cors_enabled.yml":
        assert resp.status == 200
        assert await resp.read() == b'"hi"'
        assert Headers(resp.headers)["Access-Control-Allow-Origin"] in ("*", ORIGIN)
        assert "Content-Length" in resp.headers.get("Access-Control-Expose-Headers", [])
        assert "Server" not in resp.headers.get("Access-Control-Expect-Headers", [])
    else:
        assert resp.status == 200
        assert await resp.read() == b'"hi"'
        assert resp.headers.get("Access-Control-Allow-Origin") not in ("*", ORIGIN)
        assert "Content-Length" not in resp.headers.get(
            "Access-Control-Expose-Headers", []
        )

    # a origin mismatch test
    if fname == "cors_enabled.yml":
        ORIGIN2 = "http://bentoml.ai"

        resp = await client.async_request(
            "OPTIONS",
            "/echo_json",
            headers={
                "Content-Type": "application/json",
                "Origin": ORIGIN2,
                "Access-Control-Request-Method": "POST",
                "Access-Control-Request-Headers": "Content-Type",
            },
        )

        assert resp.status != 200

        resp = await client.async_request(
            "POST",
            "/echo_json",
            headers={"Content-Type": "application/json", "Origin": ORIGIN2},
            data='"hi"',
        )

        assert resp.status == 200
        assert await resp.read() == b'"hi"'
        assert resp.headers.get("Access-Control-Allow-Origin") not in ("*", ORIGIN2)


def test_service_init_checks():
    py_model1 = bentoml.picklable_model.get("py_model.case-1.http.e2e").to_runner(
        name="invalid"
    )
    py_model2 = bentoml.picklable_model.get("py_model.case-1.http.e2e").to_runner(
        name="invalid"
    )
    with pytest.raises(ValueError) as excinfo:
        _ = bentoml.Service(name="duplicates_runners", runners=[py_model1, py_model2])
    assert "Found duplicate name" in str(excinfo.value)

    with pytest.raises(AssertionError) as excinfo:
        _ = bentoml.Service(name="invalid_model_type", models=[1])
    assert "Service models list can only" in str(excinfo.value)


def test_dunder_string():
    runner = bentoml.picklable_model.get("py_model.case-1.http.e2e").to_runner()

    svc = bentoml.Service(name="dunder_string", runners=[runner])

    assert (
        str(svc)
        == 'bentoml.Service(name="dunder_string", runners=[py_model.case-1.http.e2e])'
    )


@pytest.mark.asyncio
async def test_metrics_type(arequest: F, deployment_mode: str):
    await arequest(
        api_name="echo_data_metric",
        headers={"Content-Type": "application/json"},
        data="input_string",
        assert_output=b'"input_string"',
    )

    # The reason we have to do this is that there is no way
    # to access the metrics inside a running container.
    # This will ensure the test will pass
    await arequest(
        api_name="ensure_metrics_are_registered",
        headers={"Content-Type": "application/json"},
        data="input_string",
        assert_output=b'"input_string"',
    )
