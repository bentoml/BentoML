import typing as t

import numpy as np
import psutil
import pytest
import lightgbm as lgb

import bentoml
import bentoml.models
from bentoml.exceptions import BentoMLException
from tests.utils.helpers import assert_have_file_extension

if t.TYPE_CHECKING:
    import lightgbm as lgb  # noqa: F81

    from bentoml._internal.models import Model

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
) -> t.Callable[[t.Dict[str, t.Any], t.Dict[str, t.Any]], "Model"]:
    def _(metadata) -> "Model":
        tag = bentoml.lightgbm.save(
            TEST_MODEL_NAME,
            lightgbm_model,
            booster_params=params,
            metadata=metadata,
        )
        model = bentoml.models.get(tag)
        return model

    return _


@pytest.fixture()
def save_sklearn_proc(
    lightgbm_sklearn_model,
) -> t.Callable[[t.Dict[str, t.Any], t.Dict[str, t.Any]], "Model"]:
    def _(metadata) -> "Model":
        tag = bentoml.lightgbm.save(
            TEST_MODEL_NAME,
            lightgbm_sklearn_model,
            metadata=metadata,
        )
        model = bentoml.models.get(tag)
        return model

    return _


@pytest.fixture()
def wrong_module(lightgbm_model):
    with bentoml.models.create(
        "wrong_module",
        module=__name__,
        labels=None,
        options=None,
        context=None,
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
def test_lightgbm_save_load(metadata, save_proc):
    model = save_proc(metadata)
    assert model.info.metadata is not None
    assert_have_file_extension(model.path, ".txt")

    lgb_loaded = bentoml.lightgbm.load(
        model.tag,
    )

    assert isinstance(lgb_loaded, lgb.basic.Booster)
    assert lgb_loaded.predict(np.array([[0]])) == np.array([0.0])


@pytest.mark.parametrize("exc", [BentoMLException])
def test_lightgbm_load_exc(wrong_module, exc):
    with pytest.raises(exc):
        bentoml.lightgbm.load(wrong_module)


def test_lightgbm_runner_setup_run_batch(save_proc):
    info = save_proc(None)

    runner = bentoml.lightgbm.load_runner(info.tag)
    assert info.tag in runner.required_models
    assert runner.num_replica == 1

    assert runner.run_batch(np.array([[0]])) == np.array([0.0])
    assert isinstance(runner._model, lgb.basic.Booster)


def test_lightgbm_sklearn_save_load(save_sklearn_proc):
    info = save_sklearn_proc(None)
    assert_have_file_extension(info.path, ".pkl")

    sklearn_loaded = bentoml.lightgbm.load(
        info.tag,
    )

    assert isinstance(sklearn_loaded, lgb.LGBMClassifier)
    assert sklearn_loaded.predict(np.array([[0] * 10] * 10)).any() == np.array([0])


def test_lightgbm_sklearn_runner_setup_run_batch(save_sklearn_proc):
    info = save_sklearn_proc(None)
    runner = bentoml.lightgbm.load_runner(info.tag, infer_api_callback="predict_proba")

    assert info.tag in runner.required_models
    assert runner.num_replica == 1

    assert runner.run_batch(np.array([[0] * 10] * 10))[0][0] == 0.999999999999999
    assert isinstance(runner._model, lgb.LGBMClassifier)


@pytest.mark.gpus
def test_lightgbm_gpu_runner(save_proc):
    pass
