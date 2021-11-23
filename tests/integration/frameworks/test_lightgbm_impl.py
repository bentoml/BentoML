import os
import typing as t

import lightgbm as lgb
import numpy as np
import psutil
import pytest

import bentoml.lightgbm
import bentoml.models
from bentoml.exceptions import BentoMLException
from tests.utils.helpers import assert_have_file_extension

if t.TYPE_CHECKING:
    import lightgbm as lgb  # noqa: F81

    from bentoml._internal.models import Model, ModelStore

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
def lightgbm_sklearn_model() -> "lgb.LGBMClassifier":
    data_x = np.array([[0] * 10] * 10)
    data_y = np.array([0] * 10)

    gbm = lgb.LGBMClassifier()
    gbm.fit(data_x, data_y)

    return gbm


@pytest.fixture()
def save_proc(
    lightgbm_model,
    modelstore: "ModelStore",
) -> t.Callable[[t.Dict[str, t.Any], t.Dict[str, t.Any]], "Model"]:
    def _(metadata) -> "Model":
        tag = bentoml.lightgbm.save(
            TEST_MODEL_NAME,
            lightgbm_model,
            booster_params=params,
            metadata=metadata,
            model_store=modelstore,
        )
        model = modelstore.get(tag)
        return model

    return _


@pytest.fixture()
def save_sklearn_proc(
    lightgbm_sklearn_model,
    modelstore: "ModelStore",
) -> t.Callable[[t.Dict[str, t.Any], t.Dict[str, t.Any]], "Model"]:
    def _(metadata) -> "Model":
        tag = bentoml.lightgbm.save(
            TEST_MODEL_NAME,
            lightgbm_sklearn_model,
            metadata=metadata,
            model_store=modelstore,
        )
        model = modelstore.get(tag)
        return model

    return _


@pytest.fixture()
def wrong_module(lightgbm_model, modelstore: "ModelStore"):
    with bentoml.models.create(
        "wrong_module",
        module=__name__,
        labels=None,
        options=None,
        framework_context=None,
        metadata=None,
    ) as _model:
        lightgbm_model.save_model(_model.path_of("saved_model.txt"))
        return _model.path


@pytest.mark.parametrize(
    "metadata",
    [
        ({"acc": 0.876}),
    ],
)
def test_lightgbm_save_load(metadata, modelstore, save_proc):
    model = save_proc(metadata)
    assert model.info.metadata is not None
    assert_have_file_extension(model.path, ".txt")

    lgb_loaded = bentoml.lightgbm.load(
        model.tag,
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


def test_lightgbm_sklearn_save_load(modelstore, save_sklearn_proc):
    info = save_sklearn_proc(None)
    assert_have_file_extension(info.path, ".pkl")

    sklearn_loaded = bentoml.lightgbm.load(
        info.tag,
        model_store=modelstore,
    )

    assert isinstance(sklearn_loaded, lgb.LGBMClassifier)
    assert sklearn_loaded.predict(np.array([[0] * 10] * 10)).any() == np.array([0])


def test_lightgbm_sklearn_runner_setup_run_batch(modelstore, save_sklearn_proc):
    info = save_sklearn_proc(None)
    runner = bentoml.lightgbm.load_runner(
        info.tag, infer_api_callback="predict_proba", model_store=modelstore
    )

    assert info.tag in runner.required_models
    assert runner.num_concurrency_per_replica == psutil.cpu_count()
    assert runner.num_replica == 1

    assert runner.run_batch(np.array([[0] * 10] * 10))[0][0] == 0.999999999999999
    assert isinstance(runner._model, lgb.LGBMClassifier)


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
