from __future__ import annotations

import asyncio
import threading
from pathlib import Path

import httpx
import pytest

import bentoml
from bentoml.exceptions import BentoMLException


@bentoml.service(metrics={"enabled": False})
class RootFileService:
    @bentoml.api
    def read(self, path: Path, /) -> str:
        return path.read_text()


@pytest.fixture
def mock_http(monkeypatch):
    schema = RootFileService.schema()
    state = {"requests": [], "uploads": [], "handles": [], "outcome": "success"}

    def handler(request):
        state["requests"].append(request.url.path)
        if request.url.path == "/readyz":
            return httpx.Response(200)
        if request.url.path == "/schema.json":
            return httpx.Response(200, json=schema)
        assert request.method == "POST"
        assert request.url.path == "/read"
        assert request.headers["content-type"] == "text/plain"
        state["uploads"].append(request.content)
        if state["outcome"] == "http_error":
            return httpx.Response(503, text="model unavailable")
        if state["outcome"] == "transport_error":
            error = httpx.ConnectError("upload failed", request=request)
            state["error"] = error
            raise error
        return httpx.Response(200, text="accepted")

    # Replace only the HTTP boundary, preserving public schema discovery and
    # httpx's request-stream validation. Avoid optional environment proxy setup.
    for key in (
        "HTTP_PROXY",
        "HTTPS_PROXY",
        "ALL_PROXY",
        "http_proxy",
        "https_proxy",
        "all_proxy",
    ):
        monkeypatch.delenv(key, raising=False)
    transport = httpx.MockTransport(handler)
    for client_class in (httpx.Client, httpx.AsyncClient):
        monkeypatch.setattr(
            client_class, "_transport_for_url", lambda self, url: transport
        )
    yield state
    for stream in state["handles"]:
        stream.close()


def track_files(client, state, monkeypatch):
    original = client._file_manager.get_file

    def get_file(value):
        result = original(value)
        state["handles"].append(result[1])
        return result

    monkeypatch.setattr(client._file_manager, "get_file", get_file)
    assert client.service is None
    assert client.endpoints["read"].input_spec is None
    assert "/schema.json" in state["requests"]


@pytest.mark.parametrize("mode", ["async", "sync"])
@pytest.mark.parametrize("kind", ["path", "string"])
@pytest.mark.parametrize("outcome", ["success", "http_error", "transport_error"])
def test_root_file_upload(mock_http, monkeypatch, tmp_path, mode, kind, outcome):
    payload = b"file contents\x00" * 1000
    path = tmp_path / "document.txt"
    path.write_bytes(payload)
    value = path if kind == "path" else str(path)
    mock_http["outcome"] = outcome

    def check(client, result, failure):
        assert len(mock_http["handles"]) == 1
        assert mock_http["handles"][0].closed
        assert client._file_manager._opened_files == []
        if outcome == "success":
            if failure is not None:
                raise failure
            assert result == "accepted"
        elif outcome == "http_error":
            assert isinstance(failure, BentoMLException)
            assert failure.error_code == 503
        else:
            assert isinstance(failure, httpx.ConnectError)
            assert failure is mock_http["error"]
        assert mock_http["uploads"] == [payload]

    async def run_async():
        async with bentoml.AsyncHTTPClient("http://bento.test", token="") as client:
            track_files(client, mock_http, monkeypatch)
            result, failure = None, None
            try:
                result = await client.read(value)
            except Exception as exc:
                failure = exc
            check(client, result, failure)

    def run_sync():
        with bentoml.SyncHTTPClient("http://bento.test", token="") as client:
            track_files(client, mock_http, monkeypatch)
            result, failure = None, None
            try:
                result = client.read(value)
            except Exception as exc:
                failure = exc
            check(client, result, failure)

    if mode == "async":
        asyncio.run(run_async())
    else:
        run_sync()


def test_async_root_file_reads_off_event_loop(mock_http, monkeypatch, tmp_path):
    payload = b"file contents\x00" * 1000
    path = tmp_path / "document.txt"
    path.write_bytes(payload)
    reads = []
    loop_thread = threading.get_ident()

    async def run():
        async with bentoml.AsyncHTTPClient("http://bento.test", token="") as client:
            original = client._file_manager.get_file

            def get_file(value):
                name, stream, content_type = original(value)
                mock_http["handles"].append(stream)

                class Reader:
                    def read(self, size):
                        reads.append((size, threading.get_ident()))
                        return stream.read(size)

                return name, Reader(), content_type

            monkeypatch.setattr(client._file_manager, "get_file", get_file)
            assert await client.read(path) == "accepted"
            assert client._file_manager._opened_files == []

    asyncio.run(run())
    assert len(reads) > 2
    assert all(size == 4096 and thread != loop_thread for size, thread in reads)
    assert mock_http["uploads"] == [payload]
    assert mock_http["handles"][0].closed
