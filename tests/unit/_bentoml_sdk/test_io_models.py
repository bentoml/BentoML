from __future__ import annotations

import typing as t

from _bentoml_sdk.io_models import IODescriptor


def test_from_output_parameterized_iterator():
    def fn() -> t.Iterator[str]:
        yield "a"

    output = IODescriptor.from_output(fn)
    assert output.mime_type() == "text/plain"


def test_from_output_bare_iterator():
    def fn() -> t.Iterator:
        yield 1

    # should not raise IndexError for an unparameterized iterator
    output = IODescriptor.from_output(fn)
    assert output is not None
