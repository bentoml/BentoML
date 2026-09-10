from __future__ import annotations

import collections.abc as cabc
import typing as t

import pytest

from _bentoml_sdk.io_models import IODescriptor


@pytest.mark.parametrize(
    "return_annotation",
    [
        t.Iterator,
        t.Generator,
        t.AsyncIterator,
        t.AsyncGenerator,
        cabc.Iterator,
        cabc.Generator,
        cabc.AsyncIterator,
        cabc.AsyncGenerator,
    ],
)
def test_from_output_accepts_bare_iterators(return_annotation: t.Any) -> None:
    def stream() -> t.Iterator[t.Any]:
        yield None

    stream.__annotations__["return"] = return_annotation

    descriptor = IODescriptor.from_output(stream)

    assert descriptor.model_fields["root"].annotation is t.Any
