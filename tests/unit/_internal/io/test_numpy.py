from __future__ import annotations

import logging
from typing import TYPE_CHECKING
from functools import partial

import numpy as np
import pytest

from bentoml.io import NumpyNdarray
from bentoml.exceptions import BadInput
from bentoml.exceptions import BentoMLException
from bentoml._internal.service.openapi.specification import Schema

if TYPE_CHECKING:
    from _pytest.logging import LogCaptureFixture


class ExampleGeneric(str, np.generic):
    pass


example = np.zeros((2, 2, 3, 2))
from_example = NumpyNdarray.from_sample(example)


def test_invalid_dtype():
    with pytest.raises(BentoMLException) as e:
        NumpyNdarray(dtype="asdf")
    assert "Invalid dtype" in str(e.value)

    generic = ExampleGeneric("asdf")
    with pytest.raises(BentoMLException) as e:
        _ = NumpyNdarray.from_sample(generic)  # type: ignore (test exception)
    assert "expects a 'numpy.array'" in str(e.value)


@pytest.mark.parametrize("dtype, expected", [("float", "number"), (">U8", "integer")])
def test_numpy_to_openapi_types(dtype: str, expected: str):
    assert NumpyNdarray(dtype=dtype)._openapi_types() == expected  # type: ignore (private functions warning)


def test_numpy_openapi_schema():
    nparray = NumpyNdarray().openapi_schema()
    assert nparray.type == "array"
    assert nparray.nullable
    assert nparray.items and nparray.items.type == "integer"

    ndarray = from_example.openapi_schema()
    assert nparray.type == "array"
    assert isinstance(nparray.items, Schema)
    items = ndarray.items
    assert items.type == "array"
    assert items.items and items.items.type == "number"


def test_numpy_openapi_request_body():
    nparray = NumpyNdarray().openapi_request_body()
    assert nparray.required

    assert nparray.content
    assert "application/json" in nparray.content

    ndarray = from_example.openapi_request_body()
    assert ndarray.required
    assert ndarray.content
    assert ndarray.content["application/json"].example == example.tolist()

    nparray = NumpyNdarray(dtype="float")
    nparray.sample_input = ExampleGeneric("asdf")  # type: ignore (test exception)
    with pytest.raises(BadInput):
        nparray.openapi_example()


def test_numpy_openapi_responses():
    responses = NumpyNdarray().openapi_responses()

    assert responses.content

    assert "application/json" in responses.content
    assert not responses.content["application/json"].example


def test_verify_numpy_ndarray(caplog: LogCaptureFixture):
    partial_check = partial(
        from_example.validate_array,
        dtype=from_example._dtype,
        shape=from_example._shape,
        exception_cls=BentoMLException,
    )

    with pytest.raises(BentoMLException) as ex:
        partial_check(np.array(["asdf"]))
    assert f'Expecting ndarray of dtype "{from_example._dtype}"' in str(ex.value)

    with pytest.raises(BentoMLException) as e:
        partial_check(np.array([[1]]))
    assert f'Expecting ndarray of shape "{from_example._shape}"' in str(e.value)

    # test cases where reshape is failed
    example = NumpyNdarray.from_sample(np.ones((2, 2, 3)))
    example._enforce_shape = False
    example._enforce_dtype = False
    with caplog.at_level(logging.DEBUG):
        example.validate_array(np.array("asdf"), shape=(2, 2, 3))
    assert "Failed to reshape" in caplog.text
