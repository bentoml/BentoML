import numpy as np

from bentoml._internal.runner.utils import TypeRef


def test_typeref():

    # assert __eq__
    assert TypeRef("numpy", "ndarray") == np.ndarray
    assert TypeRef("numpy", "ndarray") == TypeRef.from_instance(
        np.random.randint([2, 3])
    )

    # evaluate
    assert TypeRef("numpy", "ndarray").evaluate() == np.ndarray
