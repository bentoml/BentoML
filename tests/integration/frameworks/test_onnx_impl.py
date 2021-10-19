import os
import typing as t

import numpy as np
import pytest

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
    with pytest, raises(exc):
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
