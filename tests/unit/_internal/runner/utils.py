import numpy as np

from bentoml._internal.types import LazyType


def test_typeref():

    # assert __eq__
    assert LazyType("numpy", "ndarray") == np.ndarray
    assert LazyType("numpy", "ndarray") == LazyType(type(np.array([2, 3])))

    # evaluate
    assert LazyType("numpy", "ndarray").get_class() == np.ndarray
