from pathlib import Path

import bentoml


def test_command_in_service_argument(examples: Path) -> None:
    with bentoml.serve(
        "service:StaticHTTP1", working_dir=str(examples / "static_http"), port=35679
    ) as server:
        with bentoml.SyncHTTPClient(server.url, server_ready_timeout=100) as client:
            resp = client.request("GET", "/test.txt")
            assert resp.status_code == 200
            assert resp.text.strip() == "Hello world!"


def test_command_in_method(examples: Path) -> None:
    with bentoml.serve(
        "service:StaticHTTP2", working_dir=str(examples / "static_http"), port=35680
    ) as server:
        with bentoml.SyncHTTPClient(server.url, server_ready_timeout=100) as client:
            resp = client.request("GET", "/test.txt")
            assert resp.status_code == 200
            assert resp.text.strip() == "Hello world!"

            resp = client.request("GET", "/metrics")
            assert resp.status_code == 200
            assert "# HELLO from custom metrics" in resp.text
