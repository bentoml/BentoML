import os
import sys
import time
import asyncio

import pytest

import bentoml
from bentoml.exceptions import BentoMLException


def test_http_server(bentoml_home: str):
    server = bentoml.HTTPServer("service.py:svc", port=12345)

    server.start()

    client = server.get_client()
    resp = client.health()

    assert resp.status == 200

    res = client.echo_json_sync({"test": "json"})

    assert res == {"test": "json"}

    server.stop()

    timeout = 10
    start_time = time.time()
    while time.time() - start_time < timeout:
        retcode = server.process.poll()
        if retcode is not None and retcode <= 0:
            break
    if sys.platform == "win32":
        # on Windows, because of the way that terminate is run, it seems the exit code is set.
        assert isinstance(server.process.poll(), int)
    else:
        # on POSIX negative return codes mean the process was terminated; since we will be terminating
        # the process, it should be negative.
        # on all other platforms, this should be 0.
        assert server.process.poll() <= 0


def test_http_server_ctx(bentoml_home: str):
    server = bentoml.HTTPServer("service.py:svc", port=12346)

    with server.start() as client:
        resp = client.health()

        assert resp.status == 200

        res = client.echo_json_sync({"more_test": "and more json"})

        assert res == {"more_test": "and more json"}

    timeout = 10
    start_time = time.time()
    while time.time() - start_time < timeout:
        retcode = server.process.poll()
        if retcode is not None and retcode <= 0:
            break
    if sys.platform == "win32":
        # on Windows, because of the way that terminate is run, it seems the exit code is set.
        assert isinstance(server.process.poll(), int)
    else:
        # on POSIX negative return codes mean the process was terminated; since we will be terminating
        # the process, it should be negative.
        # on all other platforms, this should be 0.
        assert server.process.poll() <= 0


def test_serve_from_svc():
    from service import svc

    server = bentoml.HTTPServer(svc, port=12348)
    server.start()
    client = server.get_client()
    resp = client.health()
    assert resp.status == 200
    server.stop()

    timeout = 60
    start_time = time.time()
    while time.time() - start_time < timeout:
        retcode = server.process.poll()
        if retcode is not None and retcode <= 0:
            break
    if sys.platform == "win32":
        # on Windows, because of the way that terminate is run, it seems the exit code is set.
        assert isinstance(server.process.poll(), int)
    else:
        # on POSIX negative return codes mean the process was terminated; since we will be terminating
        # the process, it should be negative.
        # on all other platforms, this should be 0.
        assert server.process.poll() <= 0


def test_serve_with_timeout(bentoml_home: str):
    server = bentoml.HTTPServer("service.py:svc", port=12349)
    config_file = os.path.abspath("configs/timeout.yml")
    env = os.environ.copy()
    env.update(BENTOML_CONFIG=config_file)

    with server.start(env=env) as client:
        with pytest.raises(
            BentoMLException,
            match="504: b'Not able to process the request in 1 seconds'",
        ):
            client.echo_delay({})


@pytest.mark.asyncio
async def test_serve_with_api_max_concurrency(bentoml_home: str):
    server = bentoml.HTTPServer("service.py:svc", port=12350, api_workers=1)
    config_file = os.path.abspath("configs/max_concurrency.yml")
    env = os.environ.copy()
    env.update(BENTOML_CONFIG=config_file)

    with server.start(env=env) as client:
        tasks = [
            asyncio.create_task(client.async_echo_delay({"delay": 0.5})),
            asyncio.create_task(client.async_echo_delay({"delay": 0.5})),
        ]
        await asyncio.sleep(0.1)
        tasks.append(asyncio.create_task(client.async_echo_delay({"delay": 0.5})))
        results = await asyncio.gather(*tasks, return_exceptions=True)

    for i in range(2):
        assert results[i] == {"delay": 0.5}, i
    assert isinstance(results[-1], BentoMLException), "unexpected success"
    assert "Too many requests" in str(results[-1]), "unexpected error message"
