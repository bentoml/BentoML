import typing as t
from typing import TYPE_CHECKING

import numpy as np
import joblib
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
if TYPE_CHECKING:
    from bentoml import Tag


def save_test_model(
    metadata: t.Dict[str, t.Any],
    labels: t.Optional[t.Dict[str, str]] = None,
    custom_objects: t.Optional[t.Dict[str, t.Any]] = None,
) -> "Tag":
    model, _ = sklearn_model_data(clf=RandomForestClassifier)
    tag_info = bentoml.sklearn.save_model(
        "test_sklearn_model",
        model,
        metadata=metadata,
        labels=labels,
        custom_objects=custom_objects,
    )
    return tag_info


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
    tag = save_test_model(metadata, labels=labels, custom_objects={"func": custom_f})
    bentomodel = bentoml.models.get(tag)
    assert bentomodel.info.metadata is not None
    assert_have_file_extension(bentomodel.path, ".pkl")
    for k in labels.keys():
        assert labels[k] == bentomodel.info.labels[k]
    assert bentomodel.custom_objects["func"](3) == custom_f(3)

    loaded = bentoml.sklearn.load_model(bentomodel.tag)

    assert isinstance(loaded, RandomForestClassifier)

    np.testing.assert_array_equal(loaded.predict(data), res_arr)


def test_sklearn_runner() -> None:
    _, data = sklearn_model_data(clf=RandomForestClassifier)
    tag = save_test_model({})
    runner = bentoml.sklearn.get(tag).to_runner()
    runner.init_local()

    assert runner.models[0].tag == tag
    assert runner.scheduled_worker_count == 1

    res = runner.run(data)
    assert (res == res_arr).all()
