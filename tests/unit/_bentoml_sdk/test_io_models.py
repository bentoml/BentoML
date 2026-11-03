from __future__ import annotations

import typing as t
from typing import Annotated

import pytest

from _bentoml_sdk.io_models import IODescriptor
from _bentoml_sdk.validators import ContentType


def sse_inner() -> t.Iterator[Annotated[str, ContentType("text/event-stream")]]:
    yield "data: chunk\n\n"


async def audio_inner() -> t.AsyncGenerator[
    Annotated[bytes, ContentType("audio/mpeg")], None
]:
    yield b""


def sse_outer() -> Annotated[t.Iterator[str], ContentType("text/event-stream")]:
    yield "data: chunk\n\n"


def xml() -> Annotated[str, ContentType("application/xml")]:
    return "<a/>"


def plain() -> str:
    return "a"


@pytest.mark.parametrize(
    "func,media_type",
    [
        (sse_inner, "text/event-stream"),
        (audio_inner, "audio/mpeg"),
        (sse_outer, "text/event-stream"),
        (xml, "application/xml"),
        (plain, None),
    ],
)
def test_from_output_media_type(
    func: t.Callable[..., t.Any], media_type: str | None
) -> None:
    desc = IODescriptor.from_output(func)
    assert desc.media_type == media_type
    if media_type is not None:
        assert desc.mime_type() == media_type
