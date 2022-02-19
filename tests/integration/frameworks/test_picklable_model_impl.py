import typing as t
from typing import TYPE_CHECKING

import numpy as np
import pytest

import bentoml
import bentoml.models

if TYPE_CHECKING:
    from bentoml._internal.store import Tag
    from bentoml._internal.models import ModelStore


class MyCoolModel:
    def predict(self, some_integer: int):
        return some_integer**2


class MyCoolBatchModel:
    def predict(self, some_integer: t.List[int]):
        return list(map(lambda x: x**2, some_integer))


class MockBatchOptions:
    enabled = True


def save_procedure(
    metadata: t.Dict[str, t.Any],
    _modelstore: "ModelStore",
) -> "Tag":
    model_to_save = MyCoolModel()
    tag_info = bentoml.picklable_model.save(
        "test_picklable_model",
        model_to_save,
        metadata=metadata,
        model_store=_modelstore,
        method="predict",
    )
    return tag_info


def save_batch_procedure(
    metadata: t.Dict[str, t.Any],
    _modelstore: "ModelStore",
) -> "Tag":
    model_to_save = MyCoolBatchModel()
    tag_info = bentoml.picklable_model.save(
        "test_picklable_model",
        model_to_save,
        batch=True,
        metadata=metadata,
        model_store=_modelstore,
        method="predict",
    )
    return tag_info


@pytest.mark.parametrize(
    "metadata",
    [({"model": "PicklableModel", "test": True})],
)
def test_picklable_model_save_load(
    metadata: t.Dict[str, t.Any],
    modelstore: "ModelStore",
) -> None:
    tag = save_procedure(metadata, _modelstore=modelstore)
    _model = bentoml.models.get(tag, _model_store=modelstore)
    assert _model.info.metadata is not None

    loaded_model = bentoml.picklable_model.load(_model.tag, model_store=modelstore)
    assert isinstance(loaded_model, MyCoolModel)
    assert loaded_model.predict(4) == np.array([16])


def test_picklable_model_runner_setup_run(modelstore: "ModelStore") -> None:

    tag = save_procedure({}, _modelstore=modelstore)
    runner = bentoml.picklable_model.load_runner(tag, model_store=modelstore)

    assert tag in runner.required_models
    assert runner.run(3) == np.array([9])


def test_pickle_runner_setup_run_method(modelstore: "ModelStore") -> None:

    tag = bentoml.picklable_model.save(
        "test_pickle_model", lambda x: x**2, metadata={}, model_store=modelstore
    )
    runner = bentoml.picklable_model.load_runner(tag, model_store=modelstore)

    assert tag in runner.required_models
    assert runner.run(3) == np.array([9])


def test_pickle_runner_setup_run_batch(modelstore: "ModelStore") -> None:

    tag = save_batch_procedure({}, _modelstore=modelstore)
    runner = bentoml.picklable_model.load_runner(tag, model_store=modelstore)

    assert tag in runner.required_models
    assert runner.run_batch([3, 9]) == [9, 81]
