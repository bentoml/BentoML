from __future__ import annotations

import typing as t
import logging
from typing import TYPE_CHECKING

import aiohttp
import multidict

logger = logging.getLogger("bentoml.tests")


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


async def async_request(
    method: str,
    url: str,
    headers: t.Union[None, t.Tuple[t.Tuple[str, str], ...], "LooseHeaders"] = None,
    data: t.Any = None,
    timeout: t.Optional[int] = None,
    assert_status: t.Union[int, t.Callable[[int], bool], None] = None,
    assert_data: t.Union[bytes, t.Callable[[bytes], bool], None] = None,
    assert_headers: t.Optional[t.Callable[[t.Any], bool]] = None,
) -> t.Tuple[int, "Headers", bytes]:
    """
    raw async request client
    """
    import aiohttp
    from starlette.datastructures import Headers

    async with aiohttp.ClientSession() as sess:
        try:
            async with sess.request(
                method, url, data=data, headers=headers, timeout=timeout
            ) as r:
                r_body = await r.read()
        except Exception:
            raise RuntimeError(
                "Unable to reach host."
            ) from None  # suppress exception trace
    if assert_status is not None:
        if callable(assert_status):
            assert assert_status(r.status), f"{r.status} {repr(r_body)}"
        else:
            assert r.status == assert_status, f"{r.status} {repr(r_body)}"

    if assert_data is not None:
        if callable(assert_data):
            assert assert_data(r_body), r_body
        else:
            assert r_body == assert_data, r_body

    if assert_headers is not None:
        assert assert_headers(r.headers), repr(r.headers)

    headers = t.cast(t.Mapping[str, str], r.headers)
    return r.status, Headers(headers), r_body


def check_headers(headers: multidict.CIMultiDict[str]) -> bool:
    return (
        headers.get("Yatai-Bento-Deployment-Name") == "sdfasdf"
        and headers.get("Yatai-Bento-Deployment-Namespace") == "yatai"
    )


async def http_proxy_app(scope: Scope, receive: Receive, send: Send):
    """
    A simplest HTTP proxy app. To simulate the behavior of yatai
    """
    if scope["type"] == "lifespan":
        return

    if scope["type"] == "http":
        async with aiohttp.ClientSession() as session:
            headers = multidict.CIMultiDict(
                tuple((k.decode(), v.decode()) for k, v in scope["headers"])
            )

            assert check_headers(headers)

            bodys: list[bytes] = []
            while True:
                request_message = await receive()
                assert request_message["type"] == "http.request"
                request_body = request_message.get("body")
                assert isinstance(request_body, bytes)
                bodys.append(request_body)
                if not request_message["more_body"]:
                    break

            async with session.request(
                method=scope["method"],
                url=scope["path"],
                headers=headers,
                data=b"".join(bodys),
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

    raise NotImplementedError(f"Scope {scope} is not understood")
