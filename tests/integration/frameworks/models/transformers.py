from __future__ import annotations

import typing as t
from typing import TYPE_CHECKING

import numpy as np
from transformers import pipeline  # type: ignore (unfinished transformers type)
from transformers.trainer_utils import set_seed

import bentoml

from . import FrameworkTestModel
from . import FrameworkTestModelInput as Input
from . import FrameworkTestModelConfiguration as Config

framework = bentoml.transformers

if TYPE_CHECKING:
    from bentoml._internal.external_typing import transformers as transformers_ext

    AnyDict = dict[str, t.Any]

tiny_model = "hf-internal-testing/tiny-random-distilbert"

set_seed(124)


def text_classification_pipeline(
    frameworks: list[str] = ["pt", "tf"]
) -> t.Generator[transformers_ext.TransformersPipeline, None, None]:
    yield from map(
        lambda framework: pipeline(model=tiny_model, framework=framework), frameworks
    )


classification_pipeline: list[FrameworkTestModel] = [
    FrameworkTestModel(
        name="text_pipeline",
        model=model,
        configurations=[
            Config(
                test_inputs={
                    "__call__": [
                        Input(
                            input_args=["i love you"],
                            expected=lambda out: np.isclose(out["score"], 0.5035),
                        )
                    ],
                },
            ),
        ],
    )
    for model in text_classification_pipeline()
]

# NOTE: when we need to add more test cases for different models
#  create a list of FrameworkTestModel and append to 'models' list
models = classification_pipeline
