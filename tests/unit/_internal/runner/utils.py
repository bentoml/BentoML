import numpy as np

from bentoml._internal.types import TypeRef


def test_typeref():

    # assert __eq__
    assert TypeRef("numpy", "ndarray") == np.ndarray
    assert TypeRef("numpy", "ndarray") == TypeRef(type(np.array([2, 3])))

    # evaluate
    assert TypeRef("numpy", "ndarray").get_class() == np.ndarray
