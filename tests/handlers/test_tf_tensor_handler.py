import sys
import json
import base64
import math
import numbers
import numpy as np

try:
    from unittest.mock import Mock, MagicMock
except ImportError:
    from mock import Mock, MagicMock


def mock_tensorflow_module():
    class MockTensor:
        def __init__(self, _input):
            self.input = _input

        def numpy(self):
            if isinstance(self.input, (list, tuple)):
                return np.array(self.input, dtype=object)
            return self.input

        def __eq__(self, dst):
            return self.input == dst.input

    class MockConstant(MockTensor):
        pass

    sys.modules['tensorflow'] = MagicMock()

    import tensorflow as tf

    tf.__version__ = "2.0"
    tf.Tensor = tf.compat.v2.Tensor = MockTensor
    tf.constant = tf.compat.v2.constant = MockConstant


mock_tensorflow_module()


STR_BYTES = b"hello world"
STR = STR_BYTES.decode("utf-8")
STR_B64 = base64.b64encode(STR_BYTES).decode()

BIN_BYTES = b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR"
BIN_B64 = base64.b64encode(BIN_BYTES).decode()

TEST_CASES = [
    {'instances': [[[1, 2]], [[3, 4]]]},
    {"instances": [[1.0, -float('inf'), float('inf')]]},
    {"instances": float('nan')},
    {"instances": {"b64": STR_B64}},
    {"instances": [{"b64": STR_B64}]},
    {"instances": {"b64": BIN_B64}},
    {"instances": [{"b64": BIN_B64}]},
]

EXPECTED_RESULTS = [
    [[[1, 2]], [[3, 4]]],
    [[1.0, -float('inf'), float('inf')]],
    float('nan'),
    STR,
    [STR],
    {"b64": BIN_B64},
    [{"b64": BIN_B64}],
]


def assert_eq_or_both_nan(x, y):
    if isinstance(x, numbers.Number) and isinstance(y, numbers.Number):
        assert math.isnan(x) and math.isnan(y) or math.isclose(x, y)
    else:
        assert x == y


def test_tf_tensor_handle_request():
    '''
    ref: https://www.tensorflow.org/tfx/serving/api_rest#request_format_2
    '''
    from bentoml.adapters import TfTensorInput

    request = Mock()
    request.headers = {}
    request.content_type = 'application/json'

    input_adapter = TfTensorInput()

    for input_data, except_result in zip(TEST_CASES, EXPECTED_RESULTS):
        request.data = json.dumps(input_data).encode('utf-8')
        response = input_adapter.handle_request(request, lambda i: i)

        prediction = json.loads(response.get_data())
        assert_eq_or_both_nan(except_result, prediction)


def test_tf_tensor_handle_batch_request():
    '''
    ref: https://www.tensorflow.org/tfx/serving/api_rest#request_format_2
    '''
    from bentoml.adapters import TfTensorInput

    input_adapter = TfTensorInput()
    request = Mock()

    for input_data, except_result in zip(TEST_CASES, EXPECTED_RESULTS):
        request.data = json.dumps(input_data).encode('utf-8')
        responses = input_adapter.handle_batch_request([request] * 3, lambda i: i)

        for response in responses:
            prediction = json.loads(response.data)
            assert_eq_or_both_nan(except_result, prediction)
