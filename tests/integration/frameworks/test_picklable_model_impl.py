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
        return some_integer ** 2

    def batch_predict(self, some_integer: t.List[int]):
        return list(map(lambda x: x ** 2, some_integer))


class MockBatchOptions:
    enabled = True


def save_procedure(
    metadata: t.Dict[str, t.Any],
    labels: t.Optional[t.Dict[str, str]] = None,
    custom_objects: t.Optional[t.Dict[str, t.Any]] = None,
) -> "Tag":
    model_to_save = MyCoolModel()
    tag = bentoml.picklable_model.save(
        "test_picklable_model",
        model_to_save,
        metadata=metadata,
        labels=labels,
        custom_objects=custom_objects,
    )
    return tag


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

    tag = save_procedure(metadata, labels=labels, custom_objects={"func": custom_f})
    bentomodel = bentoml.models.get(tag)
    assert bentomodel.info.metadata is not None
    for k in labels.keys():
        assert labels[k] == bentomodel.info.labels[k]
    assert bentomodel.custom_objects["func"](3) == custom_f(3)

    loaded_model = bentoml.picklable_model.load(bentomodel.tag)
    assert isinstance(loaded_model, MyCoolModel)
    assert loaded_model.predict(4) == np.array([16])


def test_picklable_model_runner_setup_run() -> None:

    tag = save_procedure({})
    runner = bentoml.picklable_model.load_runner(tag, method_name="predict")

    assert tag in runner.required_models
    assert runner.run(3) == np.array([9])


def test_pickle_runner_setup_run_method() -> None:
    tag = bentoml.picklable_model.save(
        "test_pickle_model", lambda x: x ** 2, metadata={}
    )
    runner = bentoml.picklable_model.load_runner(tag)

    assert tag in runner.required_models
    assert runner.run(3) == np.array([9])


def test_pickle_runner_setup_run_batch() -> None:
    tag = save_procedure({})
    runner = bentoml.picklable_model.load_runner(
        tag,
        method_name="batch_predict",
        batch=True,
    )

    assert tag in runner.required_models
    assert runner.run_batch([3, 9]) == [9, 81]
