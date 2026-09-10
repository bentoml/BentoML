from __future__ import annotations

import typing as t

from _bentoml_sdk.io_models import IODescriptor


def test_from_output_with_bare_iterator():
    def bare_iterator_fn() -> t.Iterator:
        yield 1

    descriptor = IODescriptor.from_output(bare_iterator_fn)
    assert descriptor is not None


def test_from_output_with_bare_generator():
    def bare_generator_fn() -> t.Generator:
        yield 1

    descriptor = IODescriptor.from_output(bare_generator_fn)
    assert descriptor is not None
