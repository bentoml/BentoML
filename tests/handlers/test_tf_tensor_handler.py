import pytest
from collections import namedtuple
import pandas as pd
import numpy as np
import tensorflow as tf

from bentoml.handlers import TensorflowTensorHandler

try:
    from unittest.mock import Mock
except ImportError:
    from mock import Mock


TestCase = namedtuple("TestCase", ("input", "spec", "output"))

TEST_CASES = [
    TestCase(  # row format, list of 2 tensors each of [1, 2] shape
        {'instances': [ [[1, 2]], [[3, 4]] ]},
        tf.TensorSpec(dtype=tf.int32, shape=(2, 1, 2)),
        tf.constant([ [[1, 2]], [[3, 4]] ]),
    ),
    TestCase( # with non RFC7159 values
        input={
            "instances": [[1.0, -np.Infinity, pd.Nan, np.Infinity]]
        },
        tf.TensorSpec(dtype=tf.float32, shape=(1, 4)),
        output=tf.constant([[1.0, -np.Infinity, pd.Nan, np.Infinity]], dtype=tf.float32)
    ),
    # TestCase( # with specific input name
    #     input={
    #         "instances": [
    #             {
    #                 "sensor_readings": [ 1.0, -np.Infinity, pd.Nan, np.Infinity ],
    #             }
    #         ]
    #     },
    #     tf.TensorSpec(dtype=tf.float32, shape=None),
    #     output={"predictions": [1.0, -np.Infinity, pd.Nan, np.Infinity]},
    # ),
    # TestCase( # with binary
    #     input={
    #         "signature_name": "classify_objects",
    #         "instances": [
    #             {
    #                 "image": { "b64": "aW1hZ2UgYnl0ZXM=" },
    #                 "caption": "seaside"
    #             },
    #             {
    #             "image": { "b64": "YXdlc29tZSBpbWFnZSBieXRlcw==" },
    #             "caption": "mountains"
    #             }
    #         ]
    #     },
    #     tf.TensorSpec(dtype=tf.float32, shape=None),
    #     output={},
    # ),
    # TestCase( # columnar format
    #     input={
    #         "inputs": {
    #             "tag": ["foo", "bar"],
    #             "signal": [[1, 2, 3, 4, 5], [3, 4, 1, 2, 5]],
    #             "sensor": [[[1, 2], [3, 4]], [[4, 5], [6, 8]]]
    #         }
    #     },
    #     tf.TensorSpec(dtype=tf.float32, shape=None),
    #     output={},
    # ),
    # TestCase( # List of 3 scalar tensors
    #     {'instances': ["foo", "bar", "baz"]},
    #     tf.TensorSpec(dtype=None, shape=None),
    #     tf.constant(["foo", "bar", "baz"])
    # ),
    # TestCase( # multiple named inputs
    #     input={
    #         "signature_name": "",
    #         "instances": [
    #             {
    #                 "tag": "foo",
    #                 "signal": [1, 2, 3, 4, 5],
    #                 "sensor": [[1, 2], [3, 4]]
    #             },
    #             {
    #                 "tag": "bar",
    #                 "signal": [[3, 4, 1, 2, 5]],
    #                 "sensor": [[4, 5], [6, 8]]
    #             }
    #         ]
    #     },
    #     test_function=lambda d: d,
    #     output={},
    # ),
]


def test_tf_tensor_handle_request():
    '''
    ref: https://www.tensorflow.org/tfx/serving/api_rest#request_format_2
    '''

    request = Mock()
    request.headers = {}
    request.content_type = 'application/json'

    for input_data, spec, output in TEST_CASES:
        handler = TensorflowTensorHandler(spec=spec)
        request.data = input_data
        result = handler.handle_request(request, lambda i: i)
        predictions = json.loads(result.get_data().decode('utf-8'))['predictions']
        assert predictions == output.numpy().tolist()


# def test_tf_tensor_handle_cli(capsys, tmpdir):
#     pass


# def test_tf_tensor_handle_aws_lambda_event():
#     pass
