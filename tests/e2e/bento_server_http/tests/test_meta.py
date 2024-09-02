# pylint: disable=redefined-outer-name

from __future__ import annotations

import asyncio
import json
import time
from pathlib import Path

import pytest

import bentoml
from bentoml.client import AsyncHTTPClient


@pytest.mark.asyncio
async def test_api_server_load(host: str):
    async with await AsyncHTTPClient.from_url(f"http://{host}") as client:
        for _ in range(20):
            data = json.dumps({"text": "hi"}).encode()
            response = await client.client.post(
                "/echo_json",
                data=data,
                headers={"Content-Type": "application/json"},
            )
            assert response.status_code == 200


@pytest.mark.asyncio
async def test_api_server_meta(host: str) -> None:
    async with await AsyncHTTPClient.from_url(f"http://{host}") as client:
        response = await client.client.get("/")
        assert response.status_code == 200
        response = await client.client.get("/healthz")
        assert response.status_code == 200
        response = await client.client.get("/livez")
        assert response.status_code == 200
        response = await client.client.get("/ping")
        assert response.status_code == 200
        response = await client.client.get("/hello")
        assert response.status_code == 200
        assert await response.aread() == b'{"Hello":"World"}'
        response = await client.client.get("/docs.json")
        assert response.status_code == 200
        response = await client.client.get("/metrics")
        assert response.status_code == 200
        assert await response.aread()
        response = await client.client.post("/api/v1/with_prefix")
        assert response.status_code == 404


@pytest.mark.asyncio
async def test_context(host: str):
    async with await AsyncHTTPClient.from_url(f"http://{host}") as client:
        response = await client.client.post("/use_context", params={"error": "yes"})
        assert response.status_code == 400
        assert await response.aread() == b"yes"


@pytest.mark.asyncio
async def test_runner_readiness(host: str) -> None:
    timeout = 20
    start_time = time.time()
    async with await AsyncHTTPClient.from_url(f"http://{host}") as client:
        while (time.time() - start_time) < timeout:
            response = await client.client.get("/readyz")
            if response.status_code == 200:
                break
            await asyncio.sleep(5)
        assert response.status_code == 200


@pytest.mark.asyncio
async def test_cors(host: str, server_config_file: str) -> None:
    ORIGIN = "http://bentoml.ai:8080"

    async with await AsyncHTTPClient.from_url(f"http://{host}") as client:
        headers = {
            "Content-Type": "application/json",
            "Origin": ORIGIN,
            "Access-Control-Request-Method": "POST",
            "Access-Control-Request-Headers": "Content-Type",
        }
        response = await client.client.options("/echo_json", headers=headers)

        # all test configs lives under ../configs, but we are only interested in name.
        fname = Path(server_config_file).name

        if fname == "cors_enabled.yml":
            assert response.status_code == 200
        else:
            assert response.status_code != 200

        headers = {"Content-Type": "application/json", "Origin": ORIGIN}
        response = await client.client.post("/echo_json", headers=headers, data='"hi"')

        if fname == "cors_enabled.yml":
            assert response.status_code == 200
            assert await response.aread() == b'"hi"'
            assert response.headers.get("Access-Control-Allow-Origin") in ("*", ORIGIN)
            assert "Content-Length" in response.headers.get(
                "Access-Control-Expose-Headers", []
            )
            assert "Server" not in response.headers.get(
                "Access-Control-Expect-Headers", []
            )
        else:
            assert response.status_code == 200
            assert await response.aread() == b'"hi"'
            assert response.headers.get("Access-Control-Allow-Origin") not in (
                "*",
                ORIGIN,
            )
            assert "Content-Length" not in response.headers.get(
                "Access-Control-Expose-Headers", []
            )

        # a origin mismatch test
        if fname == "cors_enabled.yml":
            ORIGIN2 = "http://bentoml.ai"

            headers = {
                "Content-Type": "application/json",
                "Origin": ORIGIN2,
                "Access-Control-Request-Method": "POST",
                "Access-Control-Request-Headers": "Content-Type",
            }
            response = await client.client.options("/echo_json", headers=headers)

            assert response.status_code != 200

            headers = {"Content-Type": "application/json", "Origin": ORIGIN2}
            response = await client.client.post(
                "/echo_json", headers=headers, data='"hi"'
            )

            assert response.status_code == 200
            assert await response.aread() == b'"hi"'
            assert response.headers.get("Access-Control-Allow-Origin") not in (
                "*",
                ORIGIN2,
            )


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
    async with await AsyncHTTPClient.from_url(f"http://{host}") as client:
        await client.client.post(
            "/echo_data_metric",
            headers={"Content-Type": "application/json"},
            data="input_string",
        )
        # The reason we have to do this is that there is no way
        # to access the metrics inside a running container.
        # This will ensure the test will pass
        response = await client.client.post(
            "/ensure_metrics_are_registered",
            headers={"Content-Type": "application/json"},
            data="input_string",
        )
        assert response.status_code == 200
