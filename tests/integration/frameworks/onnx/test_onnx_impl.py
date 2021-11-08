import os
import typing as t

import numpy as np
import pytest
import onnx

import bentoml.onnx
from bentoml.Exceptions import BentoMLException
from tests.utils.helpers import assert_have_file_extension

# fmt: on
if t.TYPE_CHECKING:
    from bentoml._internal.models.store import ModelInfo, ModelStore

TEST_MODEL_NAME = __name__.split(".")[-1]


@pytest.fixture(scope="module")
def save_proc(
    modelstore: "ModelStore",
) -> t.Callable[[t.Dict[str, t.Any], t.Dict[str, t.Any]], "ModelInfo"]:
    def _(metadata) -> "ModelInfo":
        tag = bentoml.onnx.save(
            TEST_MODEL_NAME, model, metadata=metadata, model_store=modelstore
        )
        info = modelstore.get(tag)
        return info

    return _


# Onnx conversion: https://docs.microsoft.com/en-us/windows/ai/windows-ml/onnxmltools
# Onnx models: https://docs.microsoft.com/en-us/windows/ai/windows-ml/get-onnx-model


def wrong_module(modelstore: "ModelStore"):
    # model, data =
    with modelstore.register(
        "wrong_module",
        module=__name__,
        options=None,
        metadata=None,
        framework_context=None,
    ) as ctx:
        onnx.save(model, os.path.join(ctx.path, "saved_model.onnx"))
        return str(ctx.path)


@pytest.mark.parametrize(
    "metadat",
    [
        ({"model": "Onnx", "test": True}),
        ({"acc": 0.876}),
    ],
)
def test_onnx_save_load(metadata, modelstore):  # noqa # pylint: disable
    # model, data =
    tag = bentoml.onnx.save(
        TEST_MODEL_NAME, model, metadata=metadata, model_store=modelstore
    )
    info = modelstore.get(tag)
    assert info.metadata is not None
    assert_have_file_extension(info.path, ".onnx")

    onnx_loaded = bentoml.onnx.load(tag, model_store=modelstore)

    # assert isinstance(onnx_loaded, )
    # np.testing.assert_array_equal(model.predict(data), onnx_loaded.predict(data))


@pytest.mark.parametrize("exc", [BentoMLException])
def test_get_model_info_exc(exc, modelstore):
    tag = wrong_module(modelstore)
    with pytest.raises(exc):
        bentoml.onnx._get_model_info(tag, model_store=modelstore)


def test_onnx_runner_setup_run_batch(modelstore, save_proc):
    # _, data =
    info = save_proc(None)
    runner = bentoml.onnx.load_runner(info.tag, model_store=modelstore)
    runner._setup()

    assert info.tag in runner.required_models
    # assert runner.num_concurrency_per_replica == psutil.cpu_count()
    # assert runner.num_replica ==

    res = runner._run_batch(data)
    # assert all(res, res_arr)


@pytest.mark.gpus
def test_sklearn_runner_setup_on_gpu(modelstore, save_proc):
    info = save_proc(None)
    resource_quota = dict(gpus=0, cpu=0.4)
    runner = bentoml.onnx.load_runner(
        info.tag, model_storee=modelstore, resource_quota=resource_quota
    )
    runner._setup()
    # assert runner.num_concurrency_per_replica ==
    # assert runner.num_replica ==


"""
import os

import numpy as np
import onnxruntime
import pytest
from sklearn.ensemble import RandomForestClassifier

import bentoml.onnx
from bentoml.exceptions import BentoMLException
from tests.utils.frameworks.sklearn_utils import sklearn_model_data
from tests.utils.helpers import assert_have_file_extension


def predict_arr(
    model: onnxruntime.InferenceSession,
    arr: np.array,
):
    input_data = arr.astype(np.float32)
    input_name = model.get_inputs()[0].name
    output_name = model.get_outputs()[0].name
    return model.run([output_name], {input_name: input_data})[0]


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
def test_save_raise_exc(kwargs, exc, sklearn_onnx_model, tmpdir):
    with pytest.raises(exc):
        ONNXModel.load("", **kwargs)
    with pytest.raises(exc):
        ONNXModel(sklearn_onnx_model, **kwargs).save(tmpdir)


@pytest.mark.parametrize(
    "kwargs, exc",
    [
        ({"backend": "not_supported"}, BentoMLException),
        ({"providers": ["NotSupported"]}, BentoMLException),
    ],
)
def test_load_raise_exc(kwargs, exc):
    with pytest.raises(exc):
        ONNXModel.load("", **kwargs)


def test_load_with_options(sklearn_onnx_model, tmpdir):
    _model, data = sklearn_onnx_model
    ONNXModel(_model).save(tmpdir)
    opts = onnxruntime.SessionOptions()
    opts.intra_op_num_threads = 1
    opts.inter_op_num_threads = 1
    opts.execution_mode = onnxruntime.ExecutionMode.ORT_SEQUENTIAL
    opts.log_verbosity_level = 1
    loaded = ONNXModel.load(tmpdir, sess_opts=opts)
    assert predict_arr(loaded, data)[0] == 0


def test_onnx_save_load_proto_onnxruntime(sklearn_onnx_model):
    _model, data = sklearn_onnx_model
    model: "onnxruntime.InferenceSession" = onnxruntime.InferenceSession(
        _model.SerializeToString()
    )
    onnx_loaded: "onnxruntime.InferenceSession" = ONNXModel.load(_model)
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

"""
