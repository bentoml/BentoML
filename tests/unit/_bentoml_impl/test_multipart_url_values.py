"""URL-backed values of a multipart list file field must all survive encoding.

Regression tests for the remote-schema client path: ``data[name] = url``
assigned into a mapping and kept only the last URL of a ``list[Path]`` field.
Both the httpx client builder and the aiohttp/RemoteProxy builder must emit
one repeated plain multipart field per URL, matching the wire format that a
single URL-backed value already produces.

Each request includes one real binary file so httpx selects multipart
encoding regardless of how many URL values are present.
"""

from __future__ import annotations

import aiohttp
import httpx
import pytest

from _bentoml_impl.client.base import ClientEndpoint
from _bentoml_impl.client.base import ClientFileManager
from _bentoml_impl.client.http import SyncHTTPClient as _SyncHTTPClientForType
from _bentoml_impl.client.proxy2 import AsyncClient as _ProxyAsyncClientForType

LIST_FILE_ENDPOINT = ClientEndpoint(
    name="echo",
    route="/echo",
    input={
        "properties": {
            "files": {"type": "array", "items": {"type": "file"}},
            "note": {"type": "string"},
        }
    },
)


def _make_httpx_client() -> _SyncHTTPClientForType:
    """Build an offline SyncHTTPClient exposing only what _build_multipart uses."""
    client = object.__new__(_SyncHTTPClientForType)
    client.client = httpx.Client(base_url="http://unit.test")
    client._file_manager = ClientFileManager()
    return client


def test_httpx_client_sends_every_url_of_list_file_field(tmp_path) -> None:
    binfile = tmp_path / "upload.bin"
    binfile.write_bytes(b"BIN")
    client = _make_httpx_client()
    request = client._build_multipart(  # type: ignore[arg-type]
        LIST_FILE_ENDPOINT,
        {"files": ["http://fixtures/A", "http://fixtures/B", binfile], "note": "hi"},
        httpx.Headers(),
    )
    body = request.read().decode()

    assert body.count('name="files"') == 3
    # Both URLs survive, each as a plain field (no filename), matching the
    # wire format of a single URL-backed value.
    assert body.count('name="files"; filename') == 1  # only the binary upload
    assert "http://fixtures/A" in body and "http://fixtures/B" in body
    assert body.count('name="note"') == 1


def test_httpx_client_single_url_keeps_plain_field(tmp_path) -> None:
    binfile = tmp_path / "upload.bin"
    binfile.write_bytes(b"BIN")
    client = _make_httpx_client()
    request = client._build_multipart(  # type: ignore[arg-type]
        LIST_FILE_ENDPOINT,
        {"files": ["http://fixtures/A", binfile], "note": "hi"},
        httpx.Headers(),
    )
    body = request.read().decode()

    assert body.count('name="files"') == 2
    assert body.count('name="files"; filename') == 1


@pytest.mark.asyncio
async def test_proxy_client_sends_every_url_of_list_file_field() -> None:
    proxy = object.__new__(_ProxyAsyncClientForType)  # only _file_manager is used
    proxy._file_manager = ClientFileManager()
    form = _ProxyAsyncClientForType._build_multipart(
        proxy,
        LIST_FILE_ENDPOINT,
        {"files": ["http://fixtures/A", "http://fixtures/B"], "note": "hi"},
        {},
    )

    names: list[str] = []
    values: list[str] = []
    for headers, _opts, value in form._fields:
        names.append(headers["name"])
        if "filename" not in headers:
            values.append(str(value))

    assert names.count("files") == 2
    assert "http://fixtures/A" in values
    assert "http://fixtures/B" in values
    assert isinstance(form, aiohttp.FormData)
