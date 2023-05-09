import os
import sys
import time

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
    server = bentoml.HTTPServer("service.py:svc", port=12346)
    config_file = os.path.abspath("configs/timeout.yml")
    env = os.environ.copy()
    env.update(BENTOML_CONFIG=config_file)

    with server.start(env=env) as client:
        with pytest.raises(BentoMLException, match="504: b'Not able to process the request in 1 seconds'"):
            client.echo_delay()
