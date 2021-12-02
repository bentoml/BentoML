import typing as t
import logging
from typing import TYPE_CHECKING

logger = logging.getLogger("bentoml.tests")


if TYPE_CHECKING:
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
        async with sess.request(
            method, url, data=data, headers=headers, timeout=timeout
        ) as r:
            r_body = await r.read()

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
