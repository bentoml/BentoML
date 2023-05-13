from __future__ import annotations

import typing as t
from typing import TYPE_CHECKING

import aiohttp

if TYPE_CHECKING:
    from aiohttp.typedefs import LooseHeaders
    from starlette.datastructures import FormData
    from starlette.datastructures import Headers


async def parse_multipart_form(headers: "Headers", body: bytes) -> "FormData":
    """
    parse starlette forms from headers and body
    """

    from starlette.formparsers import MultiPartParser

    async def async_bytesio(bytes_: bytes) -> t.AsyncGenerator[bytes, None]:
        yield bytes_
        yield b""
        return

    parser = MultiPartParser(headers=headers, stream=async_bytesio(body))
    return await parser.parse()


def handle_assert_exception(assert_object: t.Any, obj: t.Any, msg: str):
    res = assert_object
    try:
        if callable(assert_object):
            res = assert_object(obj)
            assert res
        else:
            assert obj == assert_object
    except AssertionError:
        raise ValueError(f"Expected: {res}. {msg}") from None
    except Exception as e:  # pylint: disable=broad-except
        # if callable has some errors, then we raise it here
        raise ValueError(
            f"Exception while excuting '{assert_object.__name__}': {e}"
        ) from None


async def async_request(
    method: str,
    url: str,
    headers: None | tuple[tuple[str, str], ...] | LooseHeaders = None,
    data: t.Any = None,
    timeout: int | None = None,
    assert_status: int | t.Callable[[int], bool] | None = None,
    assert_data: bytes | t.Callable[[bytes], bool] | None = None,
    assert_headers: t.Callable[[t.Any], bool] | None = None,
) -> tuple[int, Headers, bytes]:
    from starlette.datastructures import Headers

    async with aiohttp.ClientSession() as sess:
        try:
            async with sess.request(
                method, url, data=data, headers=headers, timeout=timeout
            ) as resp:
                body = await resp.read()
        except Exception:
            raise RuntimeError("Unable to reach host.") from None
    if assert_status is not None:
        handle_assert_exception(
            assert_status,
            resp.status,
            f"Return status [{resp.status}] with body: {body!r}",
        )
    if assert_data is not None:
        if callable(assert_data):
            msg = f"'{assert_data.__name__}' returns {assert_data(body)}"
        else:
            msg = f"Expects data '{assert_data}'"
        handle_assert_exception(
            assert_data,
            body,
            f"{msg}\nReceived response: {body}.",
        )
    if assert_headers is not None:
        handle_assert_exception(
            assert_headers,
            resp.headers,
            f"Headers assertion failed: {resp.headers!r}",
        )
    return resp.status, Headers(resp.headers), body
