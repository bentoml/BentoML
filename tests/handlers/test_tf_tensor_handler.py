import sys
import json
import base64
import math
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

COMMON_TEST_CASES = [
    {'instances': [[[1, 2]], [[3, 4]]]},
    {"instances": [[1.0, -float('inf'), float('inf')]]},
    {"instances": float('nan')},
]


def test_tf_tensor_handle_request():
    '''
    ref: https://www.tensorflow.org/tfx/serving/api_rest#request_format_2
    '''
    from bentoml.handlers import TensorflowTensorHandler

    request = Mock()
    request.headers = {}
    request.content_type = 'application/json'

    handler = TensorflowTensorHandler()

    for input_data in COMMON_TEST_CASES:
        request.data = json.dumps(input_data).encode('utf-8')
        result = handler.handle_request(request, lambda i: i)
        predictions = json.loads(result.get_data().decode('utf-8'))['predictions']
        assert (input_data['instances'] == predictions
                or math.isnan(input_data['instances']) and math.isnan(predictions))
    
    # test str b64
    input_data = {"instances": {"b64": STR_B64}}
    request.data = json.dumps(input_data).encode("utf8")
    result = handler.handle_request(request, lambda i: i)
    predictions = json.loads(result.get_data().decode('utf-8'))['predictions']
    assert STR == predictions

    input_data = {"instances": [{"b64": STR_B64}]}
    request.data = json.dumps(input_data).encode("utf8")
    result = handler.handle_request(request, lambda i: i)
    predictions = json.loads(result.get_data().decode('utf-8'))['predictions']
    assert [STR] == predictions

    # test bin b64
    input_data = {"instances": {"b64": BIN_B64}}
    request.data = json.dumps(input_data).encode("utf8")
    result = handler.handle_request(request, lambda i: i)
    predictions = json.loads(result.get_data().decode('utf-8'))['predictions']
    assert {"b64": BIN_B64} == predictions

    input_data = {"instances": [{"b64": BIN_B64}]}
    request.data = json.dumps(input_data).encode("utf8")
    result = handler.handle_request(request, lambda i: i)
    predictions = json.loads(result.get_data().decode('utf-8'))['predictions']
    assert [{"b64": BIN_B64}] == predictions

test_tf_tensor_handle_request()
