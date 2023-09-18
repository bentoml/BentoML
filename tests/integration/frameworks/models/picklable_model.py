from __future__ import annotations

import numpy as np

import bentoml

from . import FrameworkTestModel
from . import FrameworkTestModelInput as Input
from . import FrameworkTestModelConfiguration as Config

framework = bentoml.picklable_model


class PredictModel:
    def predict(self, some_integer: int):
        return some_integer**2

    def batch_predict(self, some_integer: list[int]):
        return list(map(lambda x: x**2, some_integer))


def fn(x: int) -> int:
    return x + 1


pickle_model = FrameworkTestModel(
    name="pickable_model",
    save_kwargs={
        "signatures": {
            "predict": {"batchable": False},
            "batch_predict": {"batchable": True},
        },
        "metadata": {"model": "PredictModel", "test": True},
        "custom_objects": {"func": fn},
    },
    model=PredictModel(),
    configurations=[
        Config(
            test_inputs={
                "predict": [
                    Input(input_args=[4], expected=np.array([16])),
                ],
                "batch_predict": [
                    Input(input_args=[[3, 9]], expected=[9, 81]),
                ],
            },
        ),
    ],
)

models: list[FrameworkTestModel] = [pickle_model]
