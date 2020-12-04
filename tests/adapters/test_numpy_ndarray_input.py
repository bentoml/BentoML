# pylint: disable=redefined-outer-name

import flask
import numpy as np

from bentoml.adapters import NumpyNdarrayInput

try:
    from unittest.mock import MagicMock
except ImportError:
    from mock import MagicMock


def assert_array_equal(left_array: np.ndarray, right_array: np.ndarray):
    '''
    Compare two instances of pandas.DataFrame ignoring index and columns
    '''
    try:
        if right_array.dtype == np.float:
            np.testing.assert_array_almost_equal(left_array, right_array)
        else:
            np.testing.assert_array_equal(left_array, right_array)
    except AssertionError:
        raise AssertionError(f"\n{left_array}\n is not equal to \n{right_array}\n")


def predict(ndarray):
    return ndarray * 2


def test_e2e(make_api):
    input_adapter = NumpyNdarrayInput()

    api = make_api(input_adapter, predict)

    data = b'[[1, 2, 3, 4, 5]]'
    request = MagicMock(spec=flask.Request)
    request.headers = {'Content-Type': 'application/json'}
    request.get_data.return_value = data

    result = api.handle_request(request)
    assert result.get_data().decode('utf-8') == '[[2, 4, 6, 8, 10]]'


from bentoml.types import HTTPRequest, InferenceTask


def test_from_http_request():
    input_adapter = NumpyNdarrayInput(dtype="<U21,i4,i4,i4,i4")
    request = HTTPRequest(body=b"[[1, 2, 3, 4, 5]]")
    task: InferenceTask = input_adapter.from_http_request(request)
    assert task.data == "[['1', 2, 3, 4, 5]]"


def test_extract():
    input_adapter = NumpyNdarrayInput()
    task = InferenceTask(data="[[1,2,3,4,5]]")
    args = input_adapter.extract_user_func_args([task])
    assert len(args) == 1
    assert_array_equal(args[0], np.array([[1, 2, 3, 4, 5]]))
