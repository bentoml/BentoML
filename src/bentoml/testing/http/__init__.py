from __future__ import annotations

import typing as t
import logging
import traceback

import numpy as np
import aiohttp
import multidict

import bentoml

logger = logging.getLogger(__name__)

if t.TYPE_CHECKING:
    from starlette.types import Send
    from starlette.types import Scope
    from starlette.types import Receive
    from aiohttp.typedefs import LooseHeaders
    from starlette.datastructures import Headers
    from starlette.datastructures import FormData


async def parse_multipart_form(headers: Headers, body: bytes) -> FormData:
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
    url: str,
    api_name: str,
    headers: None | tuple[tuple[str, str], ...] | LooseHeaders = None,
    data: t.Any = None,
    timeout: int | None = None,
    assert_output: t.Any | t.Callable[[t.Any], bool] | None = None,
    assert_exception: type[Exception] | tuple[type[Exception]] | None = None,
    assert_exception_match: str | t.Pattern[str] | None = None,
) -> t.Any:
    client = bentoml.client.HTTPClient.from_url(url)
    _assert_called_from_test = False

    def _assert_output(resp: t.Any):
        if assert_output is not None:
            nonlocal _assert_called_from_test
            _assert_called_from_test = True
            if callable(assert_output):
                result = assert_output(resp)
                if result is not None:
                    assert (
                        result
                    ), f"'{assert_output.__name__}' returns {result}, which is not expected."
            else:
                check = resp == assert_output
                if isinstance(check, np.ndarray):
                    check = check.all()
                assert check, f"Expects data {assert_output}, while got {resp} instead."

    try:
        if assert_exception is not None:
            try:
                import pytest
                import _pytest.outcomes
            except ImportError:
                raise bentoml.exceptions.MissingDependencyException(
                    "'pytest' is required when 'assert_exception' is not None. Make sure to install it with 'pip install -U pytest'"
                )
            try:
                with pytest.raises(assert_exception, match=assert_exception_match):
                    resp = await client.async_call(
                        api_name,
                        inp=data,
                        _http_headers=headers,
                        _http_timeout=timeout,
                    )
                    _assert_output(resp)
                    return resp
            except _pytest.outcomes.Failed:
                # In this case, the tests success, which means pytest.raises did not raise.
                pass
        else:
            resp = await client.async_call(
                api_name, inp=data, _http_headers=headers, _http_timeout=timeout
            )
            _assert_output(resp)
            return resp
    except AssertionError:
        if not _assert_called_from_test:
            raise
    except Exception as e:
        logger.error("Exception caught while sending test requests:\n")
        traceback.print_exception(type(e), e, e.__traceback__)
        raise


def assert_distributed_header(headers: multidict.CIMultiDict[str]) -> None:
    assert (
        headers.get("Yatai-Bento-Deployment-Name") == "test-deployment"
        and headers.get("Yatai-Bento-Deployment-Namespace") == "yatai"
    ), "'http_proxy_app' should only be used for simulating distributed environment similar to Yatai."


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
