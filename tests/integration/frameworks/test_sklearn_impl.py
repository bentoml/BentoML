import typing as t
from typing import TYPE_CHECKING

import joblib
import numpy as np
import psutil
import pytest
from sklearn.ensemble import RandomForestClassifier

import bentoml.models
import bentoml.sklearn
from bentoml.exceptions import BentoMLException
from tests.utils.frameworks.sklearn_utils import sklearn_model_data
from tests.utils.helpers import assert_have_file_extension
from tests.utils.types import InvalidModule, Pipeline

# fmt: off
res_arr = np.array(
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
     0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
     0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
     1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
     1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
     2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
     2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2]
)

# fmt: on
if TYPE_CHECKING:
    from bentoml._internal.models import Model, ModelStore


def save_sklearn_model(models: "Model") -> None:
    model, _ = sklearn_model_data(clf=RandomForestClassifier)
    joblib.dump(model, models.path_of("saved_model.pkl"))


@pytest.mark.parametrize(
    "metadata",
    [
        ({"model": "Sklearn", "test": True}),
        ({"acc": 0.876}),
    ],
)
def test_sklearn_save_load(
    metadata: t.Dict[str, t.Any], modelstore: "ModelStore", pipeline: Pipeline
) -> None:
    model, data = sklearn_model_data(clf=RandomForestClassifier)
    _model = pipeline(model, bentoml.sklearn, metadata=metadata)
    assert _model.info.metadata is not None
    assert_have_file_extension(_model.path, ".pkl")

    loaded = bentoml.sklearn.load(_model.tag, model_store=modelstore)

    assert isinstance(loaded, RandomForestClassifier)

    np.testing.assert_array_equal(loaded.predict(data), res_arr)


def test_get_model_info_exc(
    modelstore: "ModelStore", invalid_module: InvalidModule
) -> None:
    tag = invalid_module(save_sklearn_model)
    with pytest.raises(BentoMLException):
        bentoml.sklearn._get_model_info(tag, model_store=modelstore)


def test_sklearn_runner_setup_run_batch(
    modelstore: "ModelStore", pipeline: Pipeline
) -> None:
    model, data = sklearn_model_data(clf=RandomForestClassifier)
    _model = pipeline(model, bentoml.sklearn)
    runner = bentoml.sklearn.load_runner(_model.tag, model_store=modelstore)

    assert _model.tag in runner.required_models
    assert runner.num_concurrency_per_replica == 1
    assert runner.num_replica == psutil.cpu_count()

    res = runner.run_batch(data)
    print(res)
    assert (res == res_arr).all()
