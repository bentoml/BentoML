from __future__ import annotations

import asyncio
import io
import typing as t
from pathlib import Path

import pytest
import pytest_asyncio
from aiohttp import web

from _bentoml_impl.client.base import ClientFileManager
from _bentoml_impl.client.http import AsyncHTTPClient
from _bentoml_impl.client.http import SyncHTTPClient
from _bentoml_impl.client.proxy2 import AsyncClient

CONTENT = b"binary upload\x00\xff"


class NamedBytesIO(io.BytesIO):
    name = "data.txt"


@pytest.fixture(params=["bytesio", "named-bytesio", "buffered-file"])
def binary_stream(
    request: pytest.FixtureRequest, tmp_path: Path
) -> t.Iterator[tuple[t.BinaryIO, str, str | None]]:
    filename = "upload-file" if request.param == "bytesio" else "data.txt"
    content_type = None if request.param == "bytesio" else "text/plain"
    if request.param == "buffered-file":
        path = tmp_path / filename
        path.write_bytes(CONTENT)
        stream = path.open("rb")
    else:
        stream = (
            NamedBytesIO(CONTENT)
            if request.param == "named-bytesio"
            else io.BytesIO(CONTENT)
        )
    try:
        yield stream, filename, content_type
    finally:
        stream.close()


def test_file_manager_accepts_caller_owned_binary_stream(
    binary_stream: tuple[t.BinaryIO, str, str | None],
) -> None:
    stream, filename, content_type = binary_stream
    manager = ClientFileManager()
    try:
        assert manager.get_file(stream) == (filename, stream, content_type)
        assert stream.tell() == 0
    finally:
        manager.close()
    assert not stream.closed
    assert stream.read() == CONTENT


@pytest_asyncio.fixture
async def upload_server(unused_tcp_port: int) -> t.AsyncIterator[str]:
    async def schema(request: web.Request) -> web.Response:
        return web.json_response(
            {
                "routes": [
                    {
                        "name": "upload",
                        "route": "/upload",
                        "input": {
                            "type": "object",
                            "properties": {"file": {"type": "file"}},
                            "required": ["file"],
                        },
                        "output": {"type": "object"},
                    }
                ]
            }
        )

    async def upload(request: web.Request) -> web.Response:
        form = await request.multipart()
        file = await form.next()
        assert file is not None
        assert file.name == "file"
        return web.json_response(
            {
                "filename": file.filename,
                "content_type": file.headers.get("Content-Type"),
                "content": list(await file.read()),
            }
        )

    app = web.Application()
    app.router.add_get("/schema.json", schema)
    app.router.add_post("/upload", upload)
    server = web.AppRunner(app)
    await server.setup()
    try:
        await web.TCPSite(server, "127.0.0.1", unused_tcp_port).start()
        yield f"http://127.0.0.1:{unused_tcp_port}"
    finally:
        await server.cleanup()


@pytest.mark.asyncio
@pytest.mark.parametrize("backend", ["sync-httpx", "async-httpx", "aiohttp"])
async def test_binary_stream_multipart_upload(
    binary_stream: tuple[t.BinaryIO, str, str | None], backend: str, upload_server: str
) -> None:
    stream, filename, content_type = binary_stream
    if backend == "sync-httpx":

        def upload_sync() -> t.Any:
            with SyncHTTPClient(
                upload_server, server_ready_timeout=0, token=""
            ) as client:
                return client.call("upload", file=stream)

        result = await asyncio.to_thread(upload_sync)
    else:
        client_type = AsyncHTTPClient if backend == "async-httpx" else AsyncClient
        async with client_type(
            upload_server, server_ready_timeout=0, token=""
        ) as client:
            result = await client.call("upload", file=stream)
    assert result == {
        "filename": filename,
        "content_type": content_type or "application/octet-stream",
        "content": list(CONTENT),
    }


@pytest.mark.parametrize("as_string", [False, True])
def test_file_manager_still_owns_files_opened_from_paths(
    tmp_path: Path, as_string: bool
) -> None:
    path = tmp_path / "data.txt"
    path.write_bytes(CONTENT)
    manager = ClientFileManager()
    value = manager.get_file(str(path) if as_string else path)
    assert not isinstance(value, str)
    assert value[0] == "data.txt"
    assert value[1].read() == CONTENT
    manager.close()
    assert value[1].closed


def test_file_manager_preserves_url() -> None:
    manager = ClientFileManager()
    assert (
        manager.get_file("https://example.org/data.txt")
        == "https://example.org/data.txt"
    )
    manager.close()


@pytest.mark.parametrize("value", [None, 42, object()])
def test_file_manager_rejects_non_file_values(value: t.Any) -> None:
    with pytest.raises(AssertionError):
        ClientFileManager().get_file(value)
