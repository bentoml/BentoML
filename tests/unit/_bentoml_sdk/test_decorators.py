from __future__ import annotations

import typing

import pytest
from typing_extensions import ParamSpec

from _bentoml_sdk import decorators


@pytest.mark.parametrize("func_name", ["api", "task"])
def test_decorator_type_hints_resolve_without_stdlib_concatenate(func_name: str):
    """Regression test for #4659.

    ``decorators.py`` uses ``from __future__ import annotations``, so a
    reference like ``t.Callable[Concatenate[t.Any, P], R]`` is only resolved
    when something evaluates the string annotation, as a type checker does.
    ``typing.Concatenate`` does not exist before Python 3.10, while this
    package supports Python 3.9+, so resolving it must not depend on the
    stdlib ``typing`` module. Simulate a pre-3.10 ``typing`` module by
    removing ``Concatenate`` from it and confirm the decorator's annotations
    still resolve, proving ``Concatenate`` comes from ``typing_extensions``.

    ``P`` is swapped for a real ``ParamSpec`` for the duration of the test:
    outside of ``TYPE_CHECKING``, ``decorators.P`` is a plain ``TypeVar``
    (it only becomes a ``ParamSpec`` for type checkers), which
    ``typing.Concatenate.__getitem__`` rejects at runtime regardless of this
    fix. Using a real ``ParamSpec`` isolates the behavior under test.
    """
    func = getattr(decorators, func_name)
    original_param_spec = decorators.P
    decorators.P = ParamSpec("P")
    had_concatenate = hasattr(typing, "Concatenate")
    if had_concatenate:
        original_concatenate = typing.Concatenate
        del typing.Concatenate
    try:
        hints = typing.get_type_hints(func)
    finally:
        if had_concatenate:
            typing.Concatenate = original_concatenate
        decorators.P = original_param_spec

    assert "func" in hints
