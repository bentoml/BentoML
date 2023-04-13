# pylint: disable=redefined-outer-name

from __future__ import annotations

import time
import asyncio
from pathlib import Path

import pytest

import bentoml
from bentoml.testing.utils import async_request


@pytest.mark.asyncio
async def test_api_server_meta(host: str) -> None:
    status, _, _ = await async_request("GET", f"http://{host}/")
    assert status == 200
    status, _, _ = await async_request("GET", f"http://{host}/healthz")
    assert status == 200
    status, _, _ = await async_request("GET", f"http://{host}/livez")
    assert status == 200
    status, _, _ = await async_request("GET", f"http://{host}/ping")
    assert status == 200
    status, _, body = await async_request("GET", f"http://{host}/hello")
    assert status == 200
    assert b'{"Hello":"World"}' == body
    status, _, _ = await async_request("GET", f"http://{host}/docs.json")
    assert status == 200
    status, _, body = await async_request("GET", f"http://{host}/metrics")
    assert status == 200
    assert body
    status, _, body = await async_request("POST", f"http://{host}//api/v1/with_prefix")
    assert status == 404


@pytest.mark.asyncio
async def test_context(host: str):
    status, _, body = await async_request(
        "POST", f"http://{host}/use_context?error=yes"
    )
    assert status == 400
    assert body == b"yes"


@pytest.mark.asyncio
async def test_runner_readiness(host: str) -> None:
    timeout = 20
    start_time = time.time()
    status = ""
    while (time.time() - start_time) < timeout:
        status, _, _ = await async_request("GET", f"http://{host}/readyz")
        await asyncio.sleep(5)
        if status == 200:
            break
    assert status == 200


@pytest.mark.asyncio
async def test_cors(host: str, server_config_file: str) -> None:
    ORIGIN = "http://bentoml.ai:8080"

    status, headers, body = await async_request(
        "OPTIONS",
        f"http://{host}/echo_json",
        headers={
            "Content-Type": "application/json",
            "Origin": ORIGIN,
            "Access-Control-Request-Method": "POST",
            "Access-Control-Request-Headers": "Content-Type",
        },
    )

    # all test configs lives under ../configs, but we are only interested in name.
    fname = Path(server_config_file).name

    if fname == "cors_enabled.yml":
        assert status == 200
    else:
        assert status != 200

    status, headers, body = await async_request(
        "POST",
        f"http://{host}/echo_json",
        headers={"Content-Type": "application/json", "Origin": ORIGIN},
        data='"hi"',
    )
    if fname == "cors_enabled.yml":
        assert status == 200
        assert body == b'"hi"'
        assert headers["Access-Control-Allow-Origin"] in ("*", ORIGIN)
        assert "Content-Length" in headers.get("Access-Control-Expose-Headers", [])
        assert "Server" not in headers.get("Access-Control-Expect-Headers", [])
    else:
        assert status == 200
        assert body == b'"hi"'
        assert headers.get("Access-Control-Allow-Origin") not in ("*", ORIGIN)
        assert "Content-Length" not in headers.get("Access-Control-Expose-Headers", [])

    # a origin mismatch test
    if fname == "cors_enabled.yml":
        ORIGIN2 = "http://bentoml.ai"

        status, headers, body = await async_request(
            "OPTIONS",
            f"http://{host}/echo_json",
            headers={
                "Content-Type": "application/json",
                "Origin": ORIGIN2,
                "Access-Control-Request-Method": "POST",
                "Access-Control-Request-Headers": "Content-Type",
            },
        )

        assert status != 200

        status, headers, body = await async_request(
            "POST",
            f"http://{host}/echo_json",
            headers={"Content-Type": "application/json", "Origin": ORIGIN2},
            data='"hi"',
        )

        assert status == 200
        assert body == b'"hi"'
        assert headers.get("Access-Control-Allow-Origin") not in ("*", ORIGIN2)


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
async def test_metrics_type(host: str, deployment_mode: str):
    await async_request(
        "POST",
        f"http://{host}/echo_data_metric",
        headers={"Content-Type": "application/json"},
        data="input_string",
    )
    # The reason we have to do this is that there is no way
    # to access the metrics inside a running container.
    # This will ensure the test will pass
    await async_request(
        "POST",
        f"http://{host}/ensure_metrics_are_registered",
        headers={"Content-Type": "application/json"},
        data="input_string",
        assert_status=200,
    )
