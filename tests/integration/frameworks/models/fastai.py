from __future__ import annotations

from typing import Any
from typing import Callable
from typing import TYPE_CHECKING

import numpy as np
from fastai.torch_core import set_seed

import bentoml
from tests.utils.frameworks.fastai_utils import X
from tests.utils.frameworks.fastai_utils import SEED
from tests.utils.frameworks.fastai_utils import custom_model
from tests.utils.frameworks.fastai_utils import tabular_model

from . import FrameworkTestModel
from . import FrameworkTestModelInput as Input
from . import FrameworkTestModelConfiguration as Config

if TYPE_CHECKING:

    import bentoml._internal.external_typing as ext

framework = bentoml.fastai

set_seed(SEED, reproducible=True)


def inputs(x: list[ext.NpNDArray]) -> list[ext.NpNDArray]:
    return list(map(lambda y: y.astype(np.float32), x))


def close_to(expected: float) -> Callable[[tuple[Any, Any, ext.NpNDArray]], np.bool_]:
    def check(out: tuple[Any, Any, ext.NpNDArray]) -> np.bool_:
        return np.isclose(out[-1].squeeze().item(), expected).all()

    return check


iris_model = FrameworkTestModel(
    name="iris",
    model=tabular_model(),
    configurations=[
        Config(
            test_inputs={
                "predict": [
                    Input(
                        input_args=[X.iloc[0]],
                        expected=lambda out: np.isclose(
                            out[2].numpy(), [-0.35807556]
                        ).all(),
                    ),
                ],
            },
        ),
    ],
)

linear_regression = FrameworkTestModel(
    name="iris",
    model=custom_model(),
    configurations=[
        Config(
            test_inputs={
                "predict": [
                    Input(
                        input_args=inputs([np.array([[1] * 5])]),
                        expected=close_to(5.0),
                    ),
                ],
            },
        ),
    ],
)
models: list[FrameworkTestModel] = [iris_model, linear_regression]
