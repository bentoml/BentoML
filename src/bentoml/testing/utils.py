from __future__ import annotations

import typing as t
from typing import TYPE_CHECKING

import aiohttp
import multidict

if TYPE_CHECKING:
    from starlette.types import Send
    from starlette.types import Scope
    from starlette.types import Receive
    from aiohttp.typedefs import LooseHeaders
    from starlette.datastructures import Headers
    from starlette.datastructures import FormData


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


def assert_distributed_header(headers: multidict.CIMultiDict[str]) -> None:
    assert (
        headers.get("Yatai-Bento-Deployment-Name") == "test-deployment"
        and headers.get("Yatai-Bento-Deployment-Namespace") == "yatai"
    )


async def http_proxy_app(scope: Scope, receive: Receive, send: Send):
    """
    A simple HTTP proxy app that simulate the behavior of Yatai.
    """
    if scope["type"] == "lifespan":
        return

    if scope["type"] == "http":
        async with aiohttp.ClientSession() as session:
            headers = multidict.CIMultiDict(
                tuple((k.decode(), v.decode()) for k, v in scope["headers"])
            )

            assert_distributed_header(headers)
            bodies: list[bytes] = []
            while True:
                request_message = await receive()
                assert request_message["type"] == "http.request"
                request_body = request_message.get("body")
                assert isinstance(request_body, bytes)
                bodies.append(request_body)
                if not request_message["more_body"]:
                    break

            async with session.request(
                method=scope["method"],
                url=scope["path"],
                headers=headers,
                data=b"".join(bodies),
            ) as response:
                await send(
                    {
                        "type": "http.response.start",
                        "status": response.status,
                        "headers": list(response.raw_headers),
                    }
                )
                response_body: bytes = await response.read()
                await send(
                    {
                        "type": "http.response.body",
                        "body": response_body,
                        "more_body": False,
                    }
                )
        return

    raise NotImplementedError(f"Scope {scope} is not understood.") from None
