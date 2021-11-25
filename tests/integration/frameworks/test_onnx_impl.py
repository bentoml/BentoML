import numpy as np
import onnx
import onnxruntime as ort
import psutil
import pytest
from sklearn.ensemble import RandomForestClassifier

import bentoml.models
import bentoml.onnx
from bentoml.exceptions import BentoMLException
from tests.integration.frameworks.test_sklearn_impl import res_arr
from tests.utils.frameworks.sklearn_utils import sklearn_model_data
from tests.utils.helpers import assert_have_file_extension

TEST_MODEL_NAME = __name__.split(".")[-1]


def predict_arr(model, arr):
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


@pytest.fixture()
def save_proc(modelstore, sklearn_onnx_model):
    def _(metadata):
        model, _ = sklearn_onnx_model
        tag = bentoml.onnx.save(
            TEST_MODEL_NAME, model, metadata=metadata, model_store=modelstore
        )
        info = modelstore.get(tag)
        return info

    return _


@pytest.fixture()
def wrong_module(modelstore, sklearn_onnx_model):
    model, _ = sklearn_onnx_model
    with bentoml.models.create(
        "wrong_module",
        module=__name__,
        labels=None,
        options=None,
        metadata=None,
        framework_context=None,
    ) as _model:
        onnx.save(model, _model.path_of("saved_model.onnx"))
        return _model.path


@pytest.mark.parametrize(
    "metadata",
    [
        ({"model_type": "ONNX", "test": True}),
        ({"acc": 0.876}),
    ],
)
def test_onnx_save_load(metadata, save_proc, modelstore, sklearn_onnx_model):
    model, data = sklearn_onnx_model
    model = save_proc(metadata)
    assert model.info.metadata is not None
    assert_have_file_extension(model.path, ".onnx")

    opts = ort.SessionOptions()
    opts.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    opts.log_verbosity_level = 1
    loaded = bentoml.onnx.load(model.tag, model_store=modelstore, session_options=opts)
    assert predict_arr(loaded, data)[0] == 0


@pytest.mark.parametrize("exc", [BentoMLException])
def test_get_model_info_exc(exc, modelstore, wrong_module):
    with pytest.raises(exc):
        bentoml.onnx._get_model_info(wrong_module, model_store=modelstore)


@pytest.mark.parametrize(
    "kwargs, exc",
    [
        ({"backend": "not_supported"}, BentoMLException),
        ({"providers": ["NotSupported"]}, BentoMLException),
    ],
)
def test_load_raise_exc(kwargs, exc, modelstore, sklearn_onnx_model):
    with pytest.raises(exc):
        model, _ = sklearn_onnx_model
        tag = bentoml.onnx.save("test", model, model_store=modelstore)
        _ = bentoml.onnx.load(tag, **kwargs, model_store=modelstore)


def test_onnx_runner_setup_run_batch(modelstore, save_proc, sklearn_onnx_model):
    _, data = sklearn_onnx_model
    info = save_proc(None)
    runner = bentoml.onnx.load_runner(info.tag, model_store=modelstore)
    res = runner.run_batch(data)
    np.testing.assert_array_equal(res[0], res_arr)

    assert info.tag in runner.required_models
    assert runner.num_concurrency_per_replica == psutil.cpu_count()
    assert runner.num_replica == 1
    assert isinstance(runner._model, ort.InferenceSession)


@pytest.mark.gpus
def test_sklearn_runner_setup_on_gpu(modelstore, save_proc, sklearn_onnx_model):
    _, data = sklearn_onnx_model
    info = save_proc(None)
    runner = bentoml.onnx.load_runner(
        info.tag, model_store=modelstore, backend="onnxruntime-gpu", gpu_device_id=0
    )
    _ = runner.run_batch(data)
    assert runner.num_concurrency_per_replica == 1
    assert runner.num_replica == 1
    assert "CUDAExecutionProvider" in runner._model.get_providers()
