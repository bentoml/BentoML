import typing as t

import numpy as np
import joblib
import psutil
import pytest
from sklearn.ensemble import RandomForestClassifier

import bentoml
import bentoml.models
from bentoml.exceptions import BentoMLException
from tests.utils.helpers import assert_have_file_extension
from tests.utils.frameworks.sklearn_utils import sklearn_model_data

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
if t.TYPE_CHECKING:
    from bentoml._internal.types import Tag


def save_procedure(
    metadata: t.Dict[str, t.Any],
    labels: t.Optional[t.Dict[str, str]] = None,
    custom_objects: t.Optional[t.Dict[str, t.Any]] = None,
) -> "Tag":
    model, _ = sklearn_model_data(clf=RandomForestClassifier)
    tag_info = bentoml.sklearn.save(
        "test_sklearn_model",
        model,
        metadata=metadata,
        labels=labels,
        custom_objects=custom_objects,
    )
    return tag_info


def forbidden_procedure() -> "Tag":
    model, _ = sklearn_model_data(clf=RandomForestClassifier)
    with bentoml.models.create(
        "invalid_module",
        module=__name__,
        labels=None,
        options=None,
        context=None,
        metadata=None,
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
def test_sklearn_save_load(metadata: t.Dict[str, t.Any]) -> None:

    labels = {"stage": "dev"}

    def custom_f(x: int) -> int:
        return x + 1

    _, data = sklearn_model_data(clf=RandomForestClassifier)
    tag = save_procedure(metadata, labels=labels, custom_objects={"func": custom_f})
    bentomodel = bentoml.models.get(tag)
    assert bentomodel.info.metadata is not None
    assert_have_file_extension(bentomodel.path, ".pkl")
    for k in labels.keys():
        assert labels[k] == bentomodel.info.labels[k]
    assert bentomodel.custom_objects["func"](3) == custom_f(3)

    loaded = bentoml.sklearn.load(bentomodel.tag)

    assert isinstance(loaded, RandomForestClassifier)

    np.testing.assert_array_equal(loaded.predict(data), res_arr)


def test_get_model_info_exc() -> None:
    tag = forbidden_procedure()
    with pytest.raises(BentoMLException):
        _ = bentoml.sklearn.load(tag)


def test_sklearn_runner_setup_run_batch() -> None:
    _, data = sklearn_model_data(clf=RandomForestClassifier)
    tag = save_procedure({})
    runner = bentoml.sklearn.load_runner(tag)

    assert tag in runner.required_models
    assert runner.num_replica == 1

    res = runner.run_batch(data)
    assert (res == res_arr).all()
