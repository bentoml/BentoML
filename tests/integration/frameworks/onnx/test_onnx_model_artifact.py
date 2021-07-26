import os

import numpy as np
import onnxruntime
import pytest
from sklearn.ensemble import RandomForestClassifier

from bentoml.exceptions import BentoMLException
from bentoml.onnx import ONNXModel
from tests._internal.frameworks.sklearn_utils import sklearn_model_data
from tests._internal.helpers import assert_have_file_extension


def predict_arr(
    model: onnxruntime.InferenceSession, arr: np.array,
):
    input_data = arr.astype(np.float32)
    input_name = model.get_inputs()[0].name
    output_name = model.get_outputs()[0].name
    return model.run([output_name], {input_name: input_data})[0]


@pytest.fixture()
def sklearn_onnx_model():
    from skl2onnx import convert_sklearn
    from skl2onnx.common.data_types import FloatTensorType

    init_types = [("float_input", FloatTensorType([None, 4]))]
    model_with_data = sklearn_model_data(clf=RandomForestClassifier, num_data=30)
    return (
        convert_sklearn(model_with_data.model, initial_types=init_types),
        model_with_data.data,
    )


@pytest.mark.parametrize(
    "kwargs, exc", [({"backend": "not_supported"}, BentoMLException)]
)
def test_raise_exc(kwargs, exc, sklearn_onnx_model, tmpdir):
    with pytest.raises(exc):
        ONNXModel(sklearn_onnx_model, **kwargs).save(tmpdir)


def test_onnx_save_load_proto_onnxruntime(sklearn_onnx_model, tmpdir):
    _model, data = sklearn_onnx_model
    ONNXModel(_model).save(tmpdir)
    assert_have_file_extension(tmpdir, ".onnx")

    model: "onnxruntime.InferenceSession" = onnxruntime.InferenceSession(
        _model.SerializeToString()
    )
    onnx_loaded: "onnxruntime.InferenceSession" = ONNXModel.load(tmpdir)
    assert predict_arr(onnx_loaded, data)[0] == predict_arr(model, data)[0]


def test_onnx_save_load_filepath_onnxruntime(sklearn_onnx_model, tmpdir):
    _model, data = sklearn_onnx_model
    get_path: str = os.path.join(tmpdir, "test.onnx")
    with open(get_path, "wb") as inf:
        inf.write(_model.SerializeToString())
    model: "onnxruntime.InferenceSession" = onnxruntime.InferenceSession(
        _model.SerializeToString()
    )
    ONNXModel(get_path).save(tmpdir)
    assert_have_file_extension(tmpdir, ".onnx")

    onnx_loaded: "onnxruntime.InferenceSession" = ONNXModel.load(tmpdir)
    assert predict_arr(onnx_loaded, data)[0] == predict_arr(model, data)[0]
