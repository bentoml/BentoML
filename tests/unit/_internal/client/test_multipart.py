from __future__ import annotations

import asyncio
import typing as t
from urllib.parse import parse_qs

import httpx

from _bentoml_impl.client.base import ClientEndpoint
from _bentoml_impl.client.base import ClientFileManager
from _bentoml_impl.client.http import HTTPClient
from _bentoml_impl.client.proxy2 import AsyncClient


class _ConcreteHTTPClient(HTTPClient):
    """Concrete stand-in so the abstract client can be instantiated."""

    def _call(self, *args: t.Any, **kwargs: t.Any) -> t.Any:
        raise NotImplementedError

    def _get_stream(self, *args: t.Any, **kwargs: t.Any) -> t.Any:
        raise NotImplementedError

    def _submit(self, *args: t.Any, **kwargs: t.Any) -> t.Any:
        raise NotImplementedError


class _URLOnlyFileManager(ClientFileManager):
    """File manager whose URL values stay URL strings (no network fetch)."""

    def get_file(self, value: t.Any) -> str | tuple[str, t.IO[bytes], str | None]:
        if isinstance(value, str):
            return value
        return super().get_file(value)


def _files_endpoint() -> ClientEndpoint:
    return ClientEndpoint(
        name="echo",
        route="/echo",
        input={
            "properties": {"files": {"type": "array", "items": {"type": "file"}}},
            "required": ["files"],
        },
    )


def _http_client() -> HTTPClient:
    client = object.__new__(_ConcreteHTTPClient)
    client.client = httpx.Client()
    client._file_manager = _URLOnlyFileManager()
    return client


def _async_client() -> AsyncClient:
    client = object.__new__(AsyncClient)
    client._file_manager = _URLOnlyFileManager()
    return client


def _encode_aiohttp_form(form: t.Any) -> bytes:
    payload = form()
    value = getattr(payload, "_value", None)
    if value is not None:
        return value
    return asyncio.run(payload.read())


class TestMultipartURLListField:
    url_a = "https://files.example.invalid/a.txt"
    url_b = "https://files.example.invalid/b.txt"

    def test_httpx_client_sends_one_part_per_url(self):
        client = _http_client()
        request = client._build_multipart(
            _files_endpoint(),
            {"files": [self.url_a, self.url_b]},  # type: ignore[dict-item]
            httpx.Headers(),
        )
        body = request.read()
        assert body.count(b'name="files"') == 2, (
            "each URL in a list field must produce its own same-name part"
        )
        assert b"https://files.example.invalid/a.txt" in body
        assert b"https://files.example.invalid/b.txt" in body

    def test_httpx_client_single_url_unchanged(self):
        client = _http_client()
        request = client._build_multipart(
            _files_endpoint(),
            {"files": [self.url_a]},  # type: ignore[dict-item]
            httpx.Headers(),
        )
        assert request.read().count(b'name="files"') == 1

    def test_aiohttp_client_sends_one_part_per_url(self):
        client = _async_client()
        form = client._build_multipart(
            _files_endpoint(),
            {"files": [self.url_a, self.url_b]},  # type: ignore[dict-item]
            {"content-type": "multipart/form-data"},
        )
        body = _encode_aiohttp_form(form)
        # URL-only forms are urlencoded; the server aggregates repeated
        # field names via form.getlist, so both values must be present.
        parsed = parse_qs(body.decode())
        assert parsed["files"] == [self.url_a, self.url_b], (
            "each URL in a list field must produce its own same-name part"
        )
