from __future__ import annotations

import typing as t

from _bentoml_sdk.io_models import IODescriptor


def test_from_output_bare_iterator_does_not_raise():
    """Bare Iterator/Generator with no type args must not raise IndexError."""

    def fn() -> t.Iterator:
        yield 1

    spec = IODescriptor.from_output(fn)
    assert spec is not None
