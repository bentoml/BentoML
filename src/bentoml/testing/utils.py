from __future__ import annotations

import typing as t
from typing import TYPE_CHECKING

if TYPE_CHECKING:
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
