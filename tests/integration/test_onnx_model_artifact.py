import os

import pytest
import pandas

import bentoml
from bentoml.yatai.client import YataiClient
from tests.bento_service_examples.onnx_onnxruntime_iris_classifier import (
    OnnxIrisClassifier,
)


@pytest.fixture()
def onnx_iris_classifier_class():
    # When the ExampleBentoService got saved and loaded again in the test, the two class
    # attribute below got set to the loaded BentoService class. Resetting it here so it
    # does not effect other tests
    OnnxIrisClassifier._bento_service_bundle_path = None
    OnnxIrisClassifier._bento_service_bundle_version = None
    return OnnxIrisClassifier


@pytest.fixture()
def sklearn_onnx_model():
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


def test_onnx_model_artifact_pack_modelproto_with_onnxruntime_backend(
    onnx_iris_classifier_class, sklearn_onnx_model
):
    svc = onnx_iris_classifier_class()
    svc.pack('model', sklearn_onnx_model)
    assert svc.predict(test_df)[0] == [1], "Run inference before saving onnx artifact"

    saved_path = svc.save()
    loaded_svc = bentoml.load(saved_path)
    assert loaded_svc.predict(test_df)[0] == [1], 'Run inference after save onnx model'

    # clean up saved bundle
    yc = YataiClient()
    yc.repository.dangerously_delete_bento(svc.name, svc.version)


def test_onnx_model_artifact_pack_model_file_path_with_onnxruntime_backend(
    tmpdir, onnx_iris_classifier_class, sklearn_onnx_model
):
    model_path = os.path.join(str(tmpdir), 'test.onnx')
    with open(model_path, 'wb') as f:
        f.write(sklearn_onnx_model.SerializeToString())

    svc = onnx_iris_classifier_class()
    svc.pack('model', model_path)
    assert svc.predict(test_df)[0] == [1], "Run inference before saving onnx artifact"

    saved_path = svc.save()
    loaded_svc = bentoml.load(saved_path)
    assert loaded_svc.predict(test_df)[0] == [1], 'Run inference after save onnx model'

    # clean up saved bundle
    yc = YataiClient()
    yc.repository.dangerously_delete_bento(svc.name, svc.version)
