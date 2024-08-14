import asyncio
import os
import subprocess
import sys
from pathlib import Path

import pytest

import bentoml
from bentoml.client import AsyncHTTPClient
from bentoml.exceptions import BentoMLException


@pytest.mark.usefixtures("bentoml_home")
def test_http_server():
    server = bentoml.HTTPServer("service.py:svc", port=12345)

    server.start(stdout=sys.stdout, stderr=sys.stderr)

    client = server.get_client()
    resp = client.health()

    assert resp.status_code == 200

    res = client.call("echo_json", {"test": "json"})

    assert res == {"test": "json"}

    server.stop()

    assert server.process is not None  # process should not be removed
    try:
        retcode = server.process.wait(10)
        assert retcode is not None

        if sys.platform != "win32":
            # on Windows, because of the way that terminate is run, it seems the exit code is set.
            # negative return codes mean the process was terminated; since we will be terminating
            # the process, it should be negative.
            assert retcode <= 0
    except subprocess.TimeoutExpired:
        server.process.kill()


@pytest.mark.usefixtures("bentoml_home")
def test_http_server_ctx():
    server = bentoml.HTTPServer("service.py:svc", port=12346)

    with server.start(stdout=sys.stdout, stderr=sys.stderr) as client:
        resp = client.health()
        assert resp.status_code == 200

        res = client.call("echo_json", {"more_test": "and more json"})

        assert res == {"more_test": "and more json"}

    assert server.process is not None  # process should not be removed

    try:
        retcode = server.process.wait(10)
        assert retcode is not None

        if sys.platform != "win32":
            # on Windows, because of the way that terminate is run, it seems the exit code is set.
            # negative return codes mean the process was terminated; since we will be terminating
            # the process, it should be negative.
            assert retcode <= 0
    except subprocess.TimeoutExpired:
        server.process.kill()


def test_serve_from_svc():
    from service import svc

    server = bentoml.HTTPServer(svc, port=12348)
    server.start(stdout=sys.stdout, stderr=sys.stderr)
    client = server.get_client()
    resp = client.health()
    assert resp.status_code == 200
    server.stop()

    assert server.process is not None  # process should not be removed

    try:
        retcode = server.process.wait(10)
        assert retcode is not None

        if sys.platform != "win32":
            # on Windows, because of the way that terminate is run, it seems the exit code is set.
            # negative return codes mean the process was terminated; since we will be terminating
            # the process, it should be negative.
            assert retcode <= 0
    except subprocess.TimeoutExpired:
        server.process.kill()


@pytest.mark.usefixtures("bentoml_home")
def test_serve_with_timeout():
    server = bentoml.HTTPServer("service.py:svc", port=12349)
    config_file = os.path.abspath("configs/timeout.yml")
    env = os.environ.copy()
    env.update(BENTOML_CONFIG=config_file)

    with server.start(env=env, stdout=sys.stdout, stderr=sys.stderr) as client:
        with pytest.raises(
            BentoMLException,
            match="Not able to process the request in 1 seconds",
        ):
            client.call("echo_delay", {})


@pytest.mark.asyncio
@pytest.mark.usefixtures("bentoml_home")
async def test_serve_with_api_max_concurrency():
    server = bentoml.HTTPServer("service.py:svc", port=12350, api_workers=1)
    config_file = os.path.abspath("configs/max_concurrency.yml")
    env = os.environ.copy()
    env.update(BENTOML_CONFIG=config_file)

    with server.start(env=env, stdout=sys.stdout, stderr=sys.stderr):
        client = await bentoml.client.AsyncHTTPClient.from_url(
            f"http://{server.host}:{server.port}"
        )
        tasks = [
            asyncio.create_task(client.call("echo_delay", {"delay": 0.5})),
            asyncio.create_task(client.call("echo_delay", {"delay": 0.5})),
        ]
        await asyncio.sleep(0.1)
        tasks.append(asyncio.create_task(client.call("echo_delay", {"delay": 0.5})))
        results = await asyncio.gather(*tasks, return_exceptions=True)

    for i in range(2):
        assert results[i] == {"delay": 0.5}, i
    assert isinstance(results[-1], BentoMLException), "unexpected success"
    assert "Too many requests" in str(results[-1]), "unexpected error message"


@pytest.mark.skipif(
    os.getenv("GITHUB_ACTIONS") is not None and sys.platform == "win32",
    reason="Windows runner doesn't have enough cores to run this test",
)
@pytest.mark.asyncio
@pytest.mark.usefixtures("bentoml_home")
async def test_serve_with_lifecycle_hooks(tmp_path: Path):
    server = bentoml.HTTPServer("service.py:svc", port=12351, api_workers=4)
    env = os.environ.copy()
    env["BENTOML_TEST_DATA"] = str(tmp_path)

    with server.start(env=env, stdout=sys.stdout, stderr=sys.stderr) as client:
        assert client is not None
        async with await AsyncHTTPClient.from_url(client.server_url) as http_client:
            response = await http_client.client.post(
                "/use_context", params={"state": "data"}
            )

            assert response.status_code == 200
            assert (
                await response.aread() == b"hello"
            ), "The state data can't be read correctly"

    data_files = list(tmp_path.glob("data-*.txt"))
    assert len(data_files) == 4, "on_startup should be run 4 times"
    for f in data_files:
        assert f.read_text().strip() == "closed"

    assert (
        len(list(tmp_path.glob("deployment-*.txt"))) == 1
    ), "on_deployment should only be run once"
