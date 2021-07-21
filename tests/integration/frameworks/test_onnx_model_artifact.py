import os

import numpy as np
import onnxruntime
import pandas as pd
import pytest

from bentoml._internal.exceptions import BentoMLException
from bentoml.onnx import OnnxModel

test_df: "pd.DataFrame" = pd.DataFrame([[5.0, 4.0, 3.0, 1.0]])


def predict_df(
    model: onnxruntime.InferenceSession, df: pd.DataFrame,
):
    input_data = df.to_numpy().astype(np.float32)
    input_name = model.get_inputs()[0].name
    output_name = model.get_outputs()[0].name
    return model.run([output_name], {input_name: input_data})[0]


@pytest.fixture()
def sklearn_onnx_model():
    from skl2onnx import convert_sklearn
    from skl2onnx.common.data_types import FloatTensorType
    from sklearn.datasets import load_iris
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import train_test_split

    init_types = [('float_input', FloatTensorType([None, 4]))]

    iris = load_iris()
    X, Y = iris.data[:, :2], iris.target
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y)
    clr = RandomForestClassifier()
    clr.fit(X_train, Y_train)
    return convert_sklearn(clr, initial_types=init_types)


@pytest.mark.parametrize(
    "kwargs, exc", [({'backend': "not_supported"}, BentoMLException),]
)
def test_raise_exc(kwargs, exc, sklearn_onnx_model, tmpdir):
    with pytest.raises(exc):
        OnnxModel(sklearn_onnx_model, **kwargs).save(tmpdir)


def test_onnx_save_load_proto_onnxruntime(sklearn_onnx_model, tmpdir):
    OnnxModel(sklearn_onnx_model).save(tmpdir)
    assert os.path.exists(OnnxModel.get_path(tmpdir, ".onnx"))

    model: "onnxruntime.InferenceSession" = onnxruntime.InferenceSession(
        sklearn_onnx_model.SerializeToString()
    )
    onnx_loaded: "onnxruntime.InferenceSession" = OnnxModel.load(tmpdir)
    assert predict_df(onnx_loaded, test_df)[0] == predict_df(model, test_df)[0]


def test_onnx_save_load_filepath_onnxruntime(sklearn_onnx_model, tmpdir):
    get_path: str = os.path.join(tmpdir, 'test.onnx')
    with open(get_path, 'wb') as inf:
        inf.write(sklearn_onnx_model.SerializeToString())
    model: "onnxruntime.InferenceSession" = onnxruntime.InferenceSession(
        sklearn_onnx_model.SerializeToString()
    )
    OnnxModel(get_path).save(tmpdir)
    assert os.path.exists(OnnxModel.get_path(tmpdir, ".onnx"))

    onnx_loaded: "onnxruntime.InferenceSession" = OnnxModel.load(tmpdir)
    assert predict_df(onnx_loaded, test_df)[0] == predict_df(model, test_df)[0]
