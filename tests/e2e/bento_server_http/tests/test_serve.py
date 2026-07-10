import asyncio
import os
import sys
from pathlib import Path

import pytest

import bentoml
from bentoml.client import AsyncHTTPClient
from bentoml.client import SyncHTTPClient
from bentoml.exceptions import BentoMLException


def test_http_server():
    with bentoml.serve("service.py:svc", port=12345) as server:
        SyncHTTPClient.wait_until_server_ready("127.0.0.1", 12345)
        client = SyncHTTPClient.from_url(server.url)
        resp = client.health()

        assert resp.status_code == 200

        res = client.call("echo_json", {"test": "json"})

        assert res == {"test": "json"}
    assert not server.running


def test_http_server_ctx():
    with bentoml.serve("service.py:svc", port=12346) as server:
        SyncHTTPClient.wait_until_server_ready("127.0.0.1", 12346)
        client = SyncHTTPClient.from_url(server.url)
        resp = client.health()

        assert resp.status_code == 200

        res = client.call("echo_json", {"more_test": "and more json"})
    assert not server.running


def test_serve_from_svc():
    from service import svc

    with bentoml.serve(svc, port=12348) as server:
        SyncHTTPClient.wait_until_server_ready("127.0.0.1", 12348)
        client = SyncHTTPClient.from_url(server.url)
        resp = client.health()

        assert resp.status_code == 200

    assert not server.running


def test_serve_with_timeout(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("BENTOML_CONFIG", os.path.abspath("configs/timeout.yml"))

    with bentoml.serve("service.py:svc", port=12349) as server:
        SyncHTTPClient.wait_until_server_ready("127.0.0.1", 12349)
        client = SyncHTTPClient.from_url(server.url)
        with pytest.raises(
            BentoMLException,
            match="Not able to process the request in 1 seconds",
        ):
            client.call("echo_delay", {})


def test_serve_mounted_app_timeout(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("BENTOML_CONFIG", os.path.abspath("configs/timeout.yml"))
    # The mounted app has no BentoML APIs, so use httpx directly for the raw request.
    import httpx

    # Change into the directory that contains the service and config files so the
    # relative service path ("service_mount_timeout.py:svc") resolves correctly.
    current_dir = os.path.dirname(os.path.abspath(__file__))
    monkeypatch.chdir(os.path.join(current_dir, ".."))

    with bentoml.serve("service_mount_timeout.py:svc", port=12352) as server:
        # The mounted app is at /mounted, and the endpoint is /delay
        resp = httpx.get(f"{server.url}/mounted/delay", timeout=5)
        # The timeout middleware returns a 504 with a JSON error message
        assert resp.status_code == 504
        assert "Not able to process the request in 1 seconds" in resp.text


@pytest.mark.asyncio
async def test_serve_with_api_max_concurrency(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("BENTOML_CONFIG", os.path.abspath("configs/max_concurrency.yml"))

    with bentoml.serve("service.py:svc", port=12350, api_workers=1) as server:
        await AsyncHTTPClient.wait_until_server_ready("127.0.0.1", 12350)
        client = await AsyncHTTPClient.from_url(server.url)
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
async def test_serve_with_lifecycle_hooks(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.setenv("BENTOML_TEST_DATA", str(tmp_path))

    with bentoml.serve("service.py:svc", port=12351, api_workers=4) as server:
        await AsyncHTTPClient.wait_until_server_ready("127.0.0.1", 12351)
        async with await AsyncHTTPClient.from_url(server.url) as http_client:
            response = await http_client.client.post(
                "/use_context", params={"state": "data"}
            )

            assert response.status_code == 200
            assert await response.aread() == b"hello", (
                "The state data can't be read correctly"
            )

    data_files = list(tmp_path.glob("data-*.txt"))
    assert len(data_files) == 4, "on_startup should be run 4 times"
    for f in data_files:
        assert f.read_text().strip() == "closed"

    assert len(list(tmp_path.glob("deployment-*.txt"))) == 1, (
        "on_deployment should only be run once"
    )
