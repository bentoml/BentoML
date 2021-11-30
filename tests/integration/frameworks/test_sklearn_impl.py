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
    from bentoml._internal.types import Tag


def save_procedure(metadata: t.Dict[str, t.Any], _modelstore: "ModelStore") -> "Tag":
    model, _ = sklearn_model_data(clf=RandomForestClassifier)
    tag_info = bentoml.sklearn.save(
        "test_sklearn_model",
        model,
        metadata=metadata,
        model_store=_modelstore,
    )
    return tag_info


def forbidden_procedure(_modelstore: "ModelStore") -> "Tag":
    model, _ = sklearn_model_data(clf=RandomForestClassifier)
    with bentoml.models.create(
        "invalid_module",
        module=__name__,
        labels=None,
        options=None,
        framework_context=None,
        metadata=None,
        _model_store=_modelstore,
    ) as ctx:
        joblib.dump(model, ctx.path_of("saved_model.pkl"))
        return ctx.tag


@pytest.mark.parametrize(
    "metadata",
    [
        ({"model": "Sklearn", "test": True}),
        ({"acc": 0.876}),
    ],
)
def test_sklearn_save_load(
    metadata: t.Dict[str, t.Any], modelstore: "ModelStore"
) -> None:
    _, data = sklearn_model_data(clf=RandomForestClassifier)
    tag = save_procedure(metadata, _modelstore=modelstore)
    _model = bentoml.models.get(tag, _model_store=modelstore)
    assert _model.info.metadata is not None
    assert_have_file_extension(_model.path, ".pkl")

    loaded = bentoml.sklearn.load(_model.tag, model_store=modelstore)

    assert isinstance(loaded, RandomForestClassifier)

    np.testing.assert_array_equal(loaded.predict(data), res_arr)


def test_get_model_info_exc(modelstore: "ModelStore") -> None:
    tag = forbidden_procedure(_modelstore=modelstore)
    with pytest.raises(BentoMLException):
        _ = bentoml.sklearn.load(tag, model_store=modelstore)


def test_sklearn_runner_setup_run_batch(modelstore: "ModelStore") -> None:
    _, data = sklearn_model_data(clf=RandomForestClassifier)
    tag = save_procedure({}, _modelstore=modelstore)
    runner = bentoml.sklearn.load_runner(tag, model_store=modelstore)

    assert tag in runner.required_models
    assert runner.num_concurrency_per_replica == 1
    assert runner.num_replica == psutil.cpu_count()

    res = runner.run_batch(data)
    assert (res == res_arr).all()
