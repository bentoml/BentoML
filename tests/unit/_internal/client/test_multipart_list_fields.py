from __future__ import annotations

import io
import types
import typing as t

import httpx
import pytest

from _bentoml_impl.client.http import HTTPClient
from _bentoml_impl.client.proxy2 import AsyncClient


class _PassthroughFileManager:
    """Stands in for the real file manager.

    `ClientFileManager.get_file` returns an HTTP URL unchanged, as a plain string,
    and returns a `(filename, fileobj, content_type)` tuple for an actual upload.
    """

    def get_file(self, value: t.Any) -> t.Any:
        return value


def _endpoint(is_array: bool = True) -> types.SimpleNamespace:
    field = (
        {"type": "array", "items": {"type": "file"}} if is_array else {"type": "file"}
    )
    return types.SimpleNamespace(route="/echo", input={"properties": {"files": field}})


def _httpx_body(model: dict[str, t.Any], is_array: bool = True) -> str:
    client = types.SimpleNamespace(
        _file_manager=_PassthroughFileManager(), client=httpx.Client()
    )
    request = HTTPClient._build_multipart(
        client, _endpoint(is_array), model, httpx.Headers()
    )
    return request.read().decode("utf-8", "replace")


def _aiohttp_values(model: dict[str, t.Any]) -> list[t.Any]:
    client = types.SimpleNamespace(_file_manager=_PassthroughFileManager())
    payload = AsyncClient._build_multipart(client, _endpoint(), model, {})
    return [field[2] for field in payload._fields]


URL_A = "https://files.example.invalid/a.txt"
URL_B = "https://files.example.invalid/b.txt"


def test_httpx_client_keeps_every_url_in_a_list_field() -> None:
    # Each value of a list field resolves to a plain string, and every one of them
    # has to be sent. Assigning instead of accumulating dropped all but the last.
    body = _httpx_body({"files": [URL_A, URL_B]})

    assert "a.txt" in body
    assert "b.txt" in body


def test_aiohttp_client_keeps_every_url_in_a_list_field() -> None:
    assert _aiohttp_values({"files": [URL_A, URL_B]}) == [URL_A, URL_B]


@pytest.mark.parametrize("is_array", [True, False])
def test_single_url_is_unchanged(is_array: bool) -> None:
    # A field carrying one value must still be sent exactly once.
    model = {"files": [URL_A] if is_array else URL_A}
    assert _httpx_body(model, is_array).count("a.txt") == 1


def test_urls_and_uploads_can_be_mixed() -> None:
    # A URL travels as a form value while a real upload travels as a file part;
    # collecting the URLs must not disturb the upload.
    body = _httpx_body({"files": [URL_A, ("real.txt", io.BytesIO(b"X"), "text/plain")]})

    assert "a.txt" in body
    assert 'filename="real.txt"' in body


def test_aiohttp_single_url_is_unchanged() -> None:
    assert _aiohttp_values({"files": [URL_A]}) == [URL_A]
