import os
import typing as t

import lightgbm as lgb
import numpy as np
import psutil
import pytest

import bentoml.lightgbm
from bentoml.exceptions import BentoMLException
from tests.utils.helpers import assert_have_file_extension

if t.TYPE_CHECKING:
    from bentoml._internal.models.store import ModelInfo, ModelStore

TEST_MODEL_NAME = __name__.split(".")[-1]

params = {
    "boosting_type": "gbdt",
    "objective": "regression",
    "metric": {"l2", "l1"},
    "num_leaves": 31,
    "learning_rate": 0.05,
}


@pytest.fixture()
def lightgbm_model() -> "lgb.basic.Booster":
    data = lgb.Dataset(np.array([[0]]), label=np.array([0]))
    gbm = lgb.train(
        params,
        data,
        100,
    )

    return gbm


@pytest.fixture()
def save_proc(
    lightgbm_model,
    modelstore: "ModelStore",
) -> t.Callable[[t.Dict[str, t.Any], t.Dict[str, t.Any]], "ModelInfo"]:
    def _(metadata) -> "ModelInfo":
        tag = bentoml.lightgbm.save(
            TEST_MODEL_NAME,
            lightgbm_model,
            booster_params=params,
            metadata=metadata,
            model_store=modelstore,
        )
        info = modelstore.get(tag)
        return info

    return _


@pytest.fixture()
def wrong_module(lightgbm_model, modelstore: "ModelStore"):
    with modelstore.register(
        "wrong_module",
        module=__name__,
        options=None,
        framework_context=None,
        metadata=None,
    ) as ctx:
        lightgbm_model.save_model(os.path.join(ctx.path, "saved_model.txt"))
        return str(ctx.path)


@pytest.mark.parametrize(
    "metadata",
    [
        ({"acc": 0.876}),
    ],
)
def test_lightgbm_save_load(metadata, modelstore, save_proc):
    info = save_proc(metadata)
    assert info.metadata is not None
    assert_have_file_extension(info.path, ".txt")

    lgb_loaded = bentoml.lightgbm.load(
        info.tag,
        model_store=modelstore,
    )

    assert isinstance(lgb_loaded, lgb.basic.Booster)
    assert lgb_loaded.predict(np.array([[0]])) == np.array([0.0])


@pytest.mark.parametrize("exc", [BentoMLException])
def test_lightgbm_load_exc(wrong_module, exc, modelstore):
    with pytest.raises(exc):
        bentoml.lightgbm.load(wrong_module, model_store=modelstore)


def test_lightgbm_runner_setup_run_batch(modelstore, save_proc):
    info = save_proc(None)

    runner = bentoml.lightgbm.load_runner(info.tag, model_store=modelstore)
    assert info.tag in runner.required_models
    assert runner.num_concurrency_per_replica == psutil.cpu_count()
    assert runner.num_replica == 1

    assert runner.run_batch(np.array([[0]])) == np.array([0.0])
    assert isinstance(runner._model, lgb.basic.Booster)


@pytest.mark.gpus
def test_lightgbm_gpu_runner(modelstore, save_proc):
    booster_params = {
        "device": "gpu",
        "gpu_platform_id": 0,
        "gpu_device_id": 0,
    }
    info = save_proc(None)
    runner = bentoml.lightgbm.load_runner(
        info.tag,
        booster_params=booster_params,
        model_store=modelstore,
        resource_quota={"gpus": 0},
    )

    assert runner.num_concurrency_per_replica == 1
    assert runner.num_replica == 1
    assert info.tag in runner.required_models
    assert runner.resource_quota.on_gpu is True
