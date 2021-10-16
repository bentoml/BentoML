import os
import typing as t

import joblib
import numpy as np
import pandas as pd
import psutil
import pytest
from sklearn.ensemble import RandomForestClassifier

import bentoml.sklearn
from bentoml.exceptions import BentoMLException
from tests.utils.frameworks.sklearn_utils import sklearn_model_data
from tests.utils.helpers import assert_have_file_extension

# fmt: off
test_res_array = np.array(
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
     0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
     0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
     1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
     1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
     2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
     2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2]
)

# fmt: on

_MT = t.TypeVar("_MT")

if t.TYPE_CHECKING:
    from bentoml import ModelStore

TEST_MODEL_NAME = __name__.split(".")[-1]


@pytest.fixture(scope="module")
def save_proc(
    modelstore: "ModelStore",
) -> t.Callable[[t.Dict[str, t.Any], t.Dict[str, t.Any]], "ModelInfo"]:
    def _(metadata) -> "ModelInfo":
        model, _ = sklearn_model_data(clf=RandomForestClassifier)
        tag = bentoml.sklearn.save(
            TEST_MODEL_NAME, model, metadata=metadata, model_store=modelstore
        )
        info = modelstore.get(tag)
        return info

    return _


def wrong_module(modelstore: "ModelStore"):
    model, data = sklearn_model_data(clf=RandomForestClassifier)
    with modelstore.register(
        "wrong_module",
        module=__name__,
        options=None,
        metadata=None,
        framework_context=None,
    ) as ctx:
        joblib.dump(model, os.path.join(ctx.path, "saved_model.pkl"))
        return str(ctx.path)


@pytest.mark.parametrize(
    "metadata",
    [
        ({"model": "Sklearn", "test": True}),
        ({"acc": 0.876}),
    ],
)
def test_sklearn_save_load(metadata, modelstore):  # noqa # pylint: disable
    model, data = sklearn_model_data(clf=RandomForestClassifier)
    tag = bentoml.sklearn.save(
        TEST_MODEL_NAME, model, metadata=metadata, model_store=modelstore
    )
    info = modelstore.get(tag)
    assert info.metadata is not None
    assert_have_file_extension(info.path, ".pkl")

    sklearn_loaded = bentoml.sklearn.load(tag, model_store=modelstore)

    assert isinstance(sklearn_loaded, RandomForestClassifier)

    np.testing.assert_array_equal(
        model.predict(data), sklearn_loaded.predict(data)
    )  # noqa

    np.testing.assert_array_equal(model.predict(data), test_res_array)


@pytest.mark.parametrize("exc", [BentoMLException])
def test_get_model_info_exc(exc, modelstore):
    tag = wrong_module(modelstore)
    with pytest.raises(exc):
        bentoml.sklearn._get_model_info(tag, model_store=modelstore)


def test_sklearn_runner_setup_run_batch(modelstore, save_proc):
    _, data = sklearn_model_data()
    sklearn_params = dict()
    info = save_proc(None)
    runner = bentoml.sklearn.load_runner(info.tag, model_store=modelstore)
    runner._setup()

    assert info.tag in runner.required_models
    assert runner.num_concurrency_per_replica == psutil.cpu_count()
    assert runner.num_replica == 1

    res = runner._run_batch(data)
    assert all(res == test_res_array)

@pytest.mark.gpus
def test_sklearn_runner_setup_on_gpu(modelstore, save_proc):
    info = save_proc(None)
    resource_quota = dict(gpus=0, cpu=0.4)
    runner = bentoml.sklearn.load_runner(
        info.tag, model_store=modelstore, resource_quota=resource_quota
    )
    runner._setup()
    assert runner.num_concurrency_per_replica == 1
    assert runner.num_replica == 1


# runner.run
# runner.run_batch
