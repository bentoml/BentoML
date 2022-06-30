import typing as t
from typing import TYPE_CHECKING

import numpy as np
import pytest

import bentoml
import bentoml.models

if TYPE_CHECKING:
    from bentoml._internal.store import Tag


class MyCoolModel:
    def predict(self, some_integer: int):
        return some_integer**2

    def batch_predict(self, some_integer: t.List[int]):
        return list(map(lambda x: x**2, some_integer))


def save_test_model(
    metadata: t.Dict[str, t.Any],
    labels: t.Optional[t.Dict[str, str]] = None,
    custom_objects: t.Optional[t.Dict[str, t.Any]] = None,
) -> "Tag":
    model_to_save = MyCoolModel()
    bento_model = bentoml.picklable_model.save_model(
        "test_picklable_model",
        model_to_save,
        signatures={
            "predict": {"batchable": False},
            "batch_predict": {"batchable": True},
        },
        metadata=metadata,
        labels=labels,
        custom_objects=custom_objects,
    )
    return bento_model


@pytest.mark.parametrize(
    "metadata",
    [({"model": "PicklableModel", "test": True})],
)
def test_picklable_model_save_load(
    metadata: t.Dict[str, t.Any],
) -> None:

    labels = {"stage": "dev"}

    def custom_f(x: int) -> int:
        return x + 1

    bentomodel = save_test_model(
        metadata, labels=labels, custom_objects={"func": custom_f}
    )
    assert bentomodel.info.metadata is not None
    for k in labels.keys():
        assert labels[k] == bentomodel.info.labels[k]
    assert bentomodel.custom_objects["func"](3) == custom_f(3)

    loaded_model = bentoml.picklable_model.load_model(bentomodel.tag)
    assert isinstance(loaded_model, MyCoolModel)
    assert loaded_model.predict(4) == np.array([16])


def test_picklable_runner() -> None:
    bento_model = save_test_model({})
    runner = bento_model.to_runner()
    runner.init_local()

    assert runner.models[0].tag == bento_model.tag
    assert runner.predict.run(3) == np.array([9])
    assert runner.batch_predict.run([3, 9]) == [9, 81]


def test_picklable_model_default_signature() -> None:
    bento_model = bentoml.picklable_model.save_model(
        "test_pickle_model", lambda x: x**2, metadata={}
    )
    runner = bento_model.to_runner()
    runner.init_local()

    assert runner.models[0].tag == bento_model.tag
    assert runner.run(3) == np.array([9])
