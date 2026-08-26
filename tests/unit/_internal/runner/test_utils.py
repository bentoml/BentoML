from __future__ import annotations

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
