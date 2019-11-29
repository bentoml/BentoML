import pytest
from collecations import namedtuple
import pandas as pd
import numpy as np
import tensorflow as tf

from bentoml.handlers import TensorflowTensorHandler
from bentoml.handlers.tf_tensor_handler import check_tf_tensor_column_contains


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


def test_tf_tensor_request_schema():
    handler = TensorflowTensorHandler(
        input_dtypes={"col1": "int", "col2": "float", "col3": "string"}
    )

    schema = handler.request_schema["application/json"]["schema"]
    assert "object" == schema["type"]
    assert 3 == len(schema["properties"])
    assert "array" == schema["properties"]["col1"]["type"]
    assert "integer" == schema["properties"]["col1"]["items"]["type"]
    assert "number" == schema["properties"]["col2"]["items"]["type"]
    assert "string" == schema["properties"]["col3"]["items"]["type"]


def test_tf_tensor_handle_cli(capsys, tmpdir):
    def test_func(df):
        return df["name"][0]

    handler = TensorflowTensorHandler()

    json_file = tmpdir.join("test.json")
    with open(str(json_file), "w") as f:
        f.write('[{"name": "john","game": "mario","city": "sf"}]')

    test_args = ["--input={}".format(json_file)]
    handler.handle_cli(test_args, test_func)
    out, err = capsys.readouterr()
    assert out.strip().endswith("john")


def test_tf_tensor_handle_aws_lambda_event():
    test_content = '[{"name": "john","game": "mario","city": "sf"}]'

    def test_func(df):
        return df["name"][0]

    handler = TensorflowTensorHandler()
    success_event_obj = {
        "headers": {"Content-Type": "application/json"},
        "body": test_content,
    }
    success_response = handler.handle_aws_lambda_event(success_event_obj, test_func)

    assert success_response["statusCode"] == 200
    assert success_response["body"] == '"john"'

    error_event_obj = {
        "headers": {"Content-Type": "this_will_fail"},
        "body": test_content,
    }
    error_response = handler.handle_aws_lambda_event(error_event_obj, test_func)
    assert error_response["statusCode"] == 400


def test_check_tf_tensor_column_contains():
    df = pd.DataFrame(
        np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]]), columns=["a", "b", "c"]
    )

    # this should pass
    check_tf_tensor_column_contains({"a": "int", "b": "int", "c": "int"}, df)
    check_tf_tensor_column_contains({"a": "int"}, df)
    check_tf_tensor_column_contains({"a": "int", "c": "int"}, df)

    # this should raise exception
    with pytest.raises(ValueError) as e:
        check_tf_tensor_column_contains({"required_column_x": "int"}, df)
    assert str(e.value).startswith("Missing columns: required_column_x")

    with pytest.raises(ValueError) as e:
        check_tf_tensor_column_contains(
            {"a": "int", "b": "int", "d": "int", "e": "int"}, df
        )
    assert str(e.value).startswith("Missing columns:")


def test_tf_tensor_handle_request():
    '''
    ref: https://www.tensorflow.org/tfx/serving/api_rest#request_format_2
    '''

    handler = TensorflowTensorHandler()
    request = Mock()
    request.headers = {}
    request.content_type = 'application/json'

    # List of 3 tensors each of [1, 2] shape
    request.data = {'instances': [ [[1, 2]], [[3, 4]] ]}
    result = handler.handle_request(request, lambda d: d["instances"][0][0][0])
    assert result.get_data().decode('utf-8') == '"1"'

    # List of 3 scalar tensors.
    request.data = {'instances': ["foo", "bar", "baz"]}
    result = handler.handle_request(request, lambda d: d["instances"][0])
    assert result.get_data().decode('utf-8') == '"foo"'

    # multiple named inputs
    request.data = {
        "signature_name": "",
        "instances": [
            {
                "tag": "foo",
                "signal": [1, 2, 3, 4, 5],
                "sensor": [[1, 2], [3, 4]]
            },
            {
                "tag": "bar",
                "signal": [3, 4, 1, 2, 5]],
                "sensor": [[4, 5], [6, 8]]
            }
        ]
    }

    # non RFC7159 values 
    request.data = {
        "instances": [
            {
                "sensor_readings": [ 1.0, -Infinity, Nan, Infinity ]
            }
        ]
    }

    # column format
    request.data = {
        "inputs": {
            "tag": ["foo", "bar"],
            "signal": [[1, 2, 3, 4, 5], [3, 4, 1, 2, 5]],
            "sensor": [[[1, 2], [3, 4]], [[4, 5], [6, 8]]]
        }
    }

    # binary
    request.data = {
        "signature_name": "classify_objects",
        "examples": [
            {
            "image": { "b64": "aW1hZ2UgYnl0ZXM=" },
            "caption": "seaside"
            },
            {
            "image": { "b64": "YXdlc29tZSBpbWFnZSBieXRlcw==" },
            "caption": "mountains"
            }
        ]
    }
