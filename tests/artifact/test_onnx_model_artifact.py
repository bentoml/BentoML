import os

import pytest
import pandas

import bentoml
from bentoml.yatai.client import YataiClient
from tests.bento_service_examples.onnx_onnxruntime_iris_classifier import (
    OnnxIrisClassifier,
)


def create_sklearn_onnx_model():
    from sklearn.datasets import load_iris
    from sklearn.model_selection import train_test_split
    from sklearn.ensemble import RandomForestClassifier
    from skl2onnx import convert_sklearn
    from skl2onnx.common.data_types import FloatTensorType

    initial_types = [('float_input', FloatTensorType([None, 4]))]

    iris = load_iris()
    X, y = iris.data, iris.target
    X_train, X_test, y_train, y_test = train_test_split(X, y)
    clr = RandomForestClassifier()
    clr.fit(X_train, y_train)
    onnx_model = convert_sklearn(clr, initial_types=initial_types)
    return onnx_model


test_df = pandas.DataFrame([[5.0, 4.0, 3.0, 1.0]])


@pytest.mark.skipif(
    "TRAVIS" in os.environ and os.environ["TRAVIS"] == "true",
    reason="Skipping this test on Travis CI.",
)
def test_onnx_model_artifact_pack_modelproto_with_onnxruntime_backend():
    onnx_model = create_sklearn_onnx_model()
    svc = OnnxIrisClassifier()
    svc.pack('model', onnx_model)
    saved_path = svc.save()

    loaded_svc = bentoml.load(saved_path)
    api = loaded_svc.get_service_api('predict')

    expect_result = [1]
    result = api.func(test_df)[0]
    assert (
        expect_result == result
    ), f'Expected value {expect_result} is not equal to prediction result {result}'
    yc = YataiClient()
    yc.repository.dangerously_delete_bento(svc.name, svc.version)


@pytest.mark.skipif(
    "TRAVIS" in os.environ and os.environ["TRAVIS"] == "true",
    reason="Skipping this test on Travis CI.",
)
def test_onnx_model_artifact_pack_model_file_path_with_onnxruntime_backend(tmpdir):
    onnx_model = create_sklearn_onnx_model()
    model_path = os.path.join(str(tmpdir), 'test.onnx')
    with open(model_path, 'wb') as f:
        f.write(onnx_model.SerializeToString())
    svc = OnnxIrisClassifier()
    svc.pack('model', model_path)
    saved_path = svc.save()

    loaded_svc = bentoml.load(saved_path)
    api = loaded_svc.get_service_api('predict')

    expect_result = [1]
    result = api.func(test_df)[0]
    assert (
        expect_result == result
    ), f'Expected value {expect_result} is not equal to prediction result {result}'
    yc = YataiClient()
    yc.repository.dangerously_delete_bento(svc.name, svc.version)
