from __future__ import annotations

import typing as t
from typing import TYPE_CHECKING

import requests
import transformers
from PIL import Image
from transformers import pipeline  # type: ignore (unfinished transformers type)
from transformers.pipelines import get_task
from transformers.pipelines import SUPPORTED_TASKS
from transformers.testing_utils import (
    nested_simplify,  # type: ignore (unfinished transformers type)
)
from transformers.trainer_utils import set_seed
from transformers.pipelines.base import Pipeline
from transformers.pipelines.base import GenericTensor

import bentoml

from . import FrameworkTestModel
from . import FrameworkTestModelInput as Input
from . import FrameworkTestModelConfiguration as Config

framework = bentoml.transformers

backward_compatible = True

if TYPE_CHECKING:
    from transformers.utils.generic import ModelOutput
    from transformers.tokenization_utils_base import BatchEncoding

    from bentoml._internal.external_typing import transformers as transformers_ext

    AnyDict = dict[str, t.Any]
    AnyList = list[t.Any]
    PipelineGenerator = t.Generator[transformers_ext.TransformersPipeline, None, None]

TINY_TEXT_MODEL = "hf-internal-testing/tiny-random-distilbert"
TINY_TEXT_TASK = get_task(TINY_TEXT_MODEL)
TINY_TEXT_AUTO = "AutoModelForSequenceClassification"

set_seed(124)


class CustomPipeline(Pipeline):
    def _sanitize_parameters(self, **kwargs: t.Any) -> tuple[AnyDict, AnyDict, AnyDict]:
        preprocess_kwargs: AnyDict = {}
        if "dummy_arg" in kwargs:
            preprocess_kwargs["dummy_arg"] = kwargs["dummy_arg"]
        return preprocess_kwargs, {}, {}

    def preprocess(self, text: str, dummy_arg: int = 2) -> BatchEncoding | None:
        if self.tokenizer:
            input_ids = self.tokenizer(text, return_tensors="pt")
            return input_ids

    def _forward(
        self, input_tensors: dict[str, GenericTensor], **parameters: AnyDict
    ) -> ModelOutput:
        return self.model(**input_tensors)  # type: ignore

    def postprocess(self, model_outputs: ModelOutput, **parameters: AnyDict) -> t.Any:
        return model_outputs["logits"].softmax(-1).numpy()  # type: ignore (unfinished transformers type)


def gen_kwargs(
    task_name: str, model_name: str, model_auto: str, task_type: str
) -> AnyDict:
    task_definition = {
        "impl": CustomPipeline,
        **{
            "task": task_name,
            "tf": (),
            "pt": (model_auto,),
            "default": {"pt": (model_name,)},
            "type": task_type,
        },
    }
    return {"task_name": task_name, "task_definition": task_definition}


def gen_task_pipeline(
    model: str, task: str | None = None, *, frameworks: list[str] = ["pt", "tf"]
) -> PipelineGenerator:
    yield from map(
        lambda framework: pipeline(task=task, model=model, framework=framework),  # type: ignore (unfinished transformers type)
        frameworks,
    )


def expected_equal(
    expected: list[AnyDict | AnyList],
) -> t.Callable[[list[AnyDict | AnyList]], bool]:
    def check_output(out: list[AnyDict | AnyList]) -> bool:
        assert nested_simplify(out, decimals=4) == expected
        return True

    return check_output


text_classification_pipeline: list[FrameworkTestModel] = [
    FrameworkTestModel(
        name="text_pipeline",
        model=model,
        configurations=[
            Config(
                load_kwargs={"task": TINY_TEXT_TASK},
                test_inputs={
                    "__call__": [
                        Input(
                            input_args=["i love you"],
                            expected=expected_equal(
                                [{"label": "LABEL_0", "score": 0.5036}]
                            ),
                        )
                    ],
                },
            ),
        ],
    )
    for model in gen_task_pipeline(model=TINY_TEXT_MODEL)
]

batched_pipeline: list[FrameworkTestModel] = [
    FrameworkTestModel(
        name="batched_pipeline",
        model=model,
        save_kwargs={
            "signatures": {
                "__call__": {"batchable": True},
            }
        },
        configurations=[
            Config(
                load_kwargs={"task": TINY_TEXT_TASK},
                test_inputs={
                    "__call__": [
                        Input(
                            input_args=[["A bento box is"]],
                            expected=expected_equal(
                                [{"label": "LABEL_0", "score": 0.5035}]
                            ),
                        ),
                        Input(
                            input_args=[["This is another test"]],
                            expected=expected_equal(
                                [{"label": "LABEL_0", "score": 0.5035}]
                            ),
                        ),
                    ]
                },
            )
        ],
    )
    for model in gen_task_pipeline(model=TINY_TEXT_MODEL)
]

tiny_image_model = "hf-internal-testing/tiny-random-vit"
tiny_image_task = get_task(tiny_image_model)
tiny_image_auto = "AutoModelForImageClassification"
test_url = "http://images.cocodataset.org/val2017/000000039769.jpg"

image_classification: list[FrameworkTestModel] = [
    FrameworkTestModel(
        name="image_pipeline",
        model=model,
        configurations=[
            Config(
                load_kwargs={"task": tiny_image_task},
                test_inputs={
                    "__call__": [
                        Input(
                            input_args=[
                                [
                                    test_url,
                                    Image.open(requests.get(test_url, stream=True).raw),
                                ]
                            ],
                            expected=expected_equal(
                                [
                                    [
                                        {"label": "LABEL_1", "score": 0.574},
                                        {"label": "LABEL_0", "score": 0.426},
                                    ],
                                    [
                                        {"label": "LABEL_1", "score": 0.574},
                                        {"label": "LABEL_0", "score": 0.426},
                                    ],
                                ]
                            ),
                        )
                    ],
                },
            ),
        ],
    )
    for model in gen_task_pipeline(model=tiny_image_model)
]

custom_task = "custom-text-classification"


# save_kwargs
def create_save_kwargs() -> AnyDict:
    save_kwargs = gen_kwargs(custom_task, TINY_TEXT_MODEL, TINY_TEXT_AUTO, "text")
    save_kwargs["task_definition"]["pt"] = (
        getattr(transformers, save_kwargs["task_definition"]["pt"][0]),
    )

    return save_kwargs


# inject custom task to SUPPORTED_TASKS
SUPPORTED_TASKS[custom_task] = create_save_kwargs()["task_definition"]

custom_kwargs = gen_kwargs(custom_task, TINY_TEXT_MODEL, TINY_TEXT_AUTO, "text")
load_kwargs = custom_kwargs["task_definition"]
load_kwargs.pop("impl")


def check_model(model: transformers_ext.TransformersPipeline, dct: AnyDict) -> None:
    assert custom_task in SUPPORTED_TASKS
    assert SUPPORTED_TASKS[custom_task] == create_save_kwargs()["task_definition"]


custom_pipeline: list[FrameworkTestModel] = [
    FrameworkTestModel(
        name="custom_pipeline",
        model=model,
        save_kwargs=create_save_kwargs(),
        configurations=[
            Config(
                load_kwargs=load_kwargs,
                test_inputs={
                    "__call__": [
                        Input(
                            input_args=["i love you"],
                            expected=expected_equal([[0.504, 0.496]]),
                        )
                    ],
                },
                check_model=check_model,
            ),
        ],
    )
    for model in gen_task_pipeline(
        model=TINY_TEXT_MODEL, task=custom_task, frameworks=["pt"]
    )
]

# NOTE: when we need to add more test cases for different models
#  create a list of FrameworkTestModel and append to 'models' list
models = (
    text_classification_pipeline
    + batched_pipeline
    + image_classification
    + custom_pipeline
)
