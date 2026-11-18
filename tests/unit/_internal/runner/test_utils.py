from __future__ import annotations

import typing as t

import pytest

from bentoml._internal.runner.utils import Params


def test_params_sample_returns_first_arg():
    params = Params(1, 2, foo=3)
    assert params.sample == 1


def test_params_sample_falls_back_to_first_kwarg_when_no_args():
    params = Params(foo=3, bar=4)
    assert params.sample == 3


def test_params_sample_on_empty_params_raises_value_error():
    # A runner method that takes zero arguments produces an empty ``Params``.
    # ``sample`` must not leak a bare ``StopIteration`` (see issue #4263), which
    # is both a confusing error and a PEP 479 hazard. It should raise a clear,
    # actionable ``ValueError`` instead.
    params: Params[object] = Params()
    with pytest.raises(ValueError):
        _ = params.sample


def test_params_all_equal_true_for_matching_values():
    assert Params(1, 1, foo=1).all_equal() is True


def test_params_all_equal_false_for_differing_values():
    assert Params(1, 2).all_equal() is False


def test_params_all_equal_on_empty_params_is_vacuously_true():
    # An empty ``Params`` (zero-argument runner method) has no values that could
    # disagree, so ``all_equal`` is vacuously true. It must not leak a bare
    # ``StopIteration`` from ``next()`` on the empty iterator (see issue #4263).
    params: Params[object] = Params()
    assert params.all_equal() is True


def test_params_sample_error_message_is_actionable():
    # The point of #4263 is not merely "some exception" -- a bare
    # ``StopIteration`` was already an exception. It is that the caller is told
    # what went wrong and what to do, so an empty message would not fix the
    # reported problem.
    params: Params[object] = Params()
    with pytest.raises(ValueError) as excinfo:
        _ = params.sample

    message = str(excinfo.value)
    assert "empty Params" in message
    assert "at least one argument" in message


def test_params_sample_suppresses_the_stopiteration_context():
    # ``raise ... from None`` sets ``__suppress_context__``, which keeps the
    # original ``StopIteration`` out of the rendered traceback. Without it the
    # confusing exception this issue is about is still shown to the user, just
    # underneath a clearer one.
    params: Params[object] = Params()
    with pytest.raises(ValueError) as excinfo:
        _ = params.sample

    assert excinfo.value.__suppress_context__ is True
    assert excinfo.value.__cause__ is None


def test_params_sample_inside_a_generator_raises_value_error_not_runtime_error():
    """The concrete PEP 479 hazard behind #4263.

    A bare ``StopIteration`` escaping into a generator frame is converted by
    the interpreter into ``RuntimeError: generator raised StopIteration``,
    which names neither ``Params`` nor the real problem. Runner code calls
    ``sample`` from generator and async frames, so this is the shape the bug
    actually took in the wild.
    """

    def consume() -> t.Iterator[object]:
        params: Params[object] = Params()
        yield params.sample

    with pytest.raises(ValueError) as excinfo:
        list(consume())

    assert "empty Params" in str(excinfo.value)


def test_params_all_equal_inside_a_generator_does_not_raise():
    def consume() -> t.Iterator[bool]:
        params: Params[object] = Params()
        yield params.all_equal()

    assert list(consume()) == [True]
