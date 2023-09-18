from __future__ import annotations

import typing as t
import importlib

import numpy as np
import requests
import tensorflow as tf
import transformers
from PIL import Image
from datasets import load_dataset
from transformers import Pipeline
from transformers.utils import is_tf_available
from transformers.utils import is_torch_available
from transformers.pipelines import get_task
from transformers.pipelines import get_supported_tasks
from transformers.testing_utils import nested_simplify
from transformers.trainer_utils import set_seed

import bentoml
from bentoml._internal.frameworks.transformers import TaskDefinition
from bentoml._internal.frameworks.transformers import delete_pipeline
from bentoml._internal.frameworks.transformers import register_pipeline
from bentoml._internal.frameworks.transformers import convert_to_autoclass
from bentoml._internal.frameworks.transformers import SimpleDefaultMapping

from . import FrameworkTestModel as Model
from . import FrameworkTestModelInput as Input
from . import FrameworkTestModelConfiguration as Config

if t.TYPE_CHECKING:
    import torch
    from numpy.typing import NDArray
    from transformers.utils.generic import ModelOutput
    from transformers.pipelines.base import GenericTensor
    from transformers.tokenization_utils_base import BatchEncoding

    from bentoml._internal.external_typing import transformers as transformers_ext

    AnyDict = dict[str, t.Any]
    AnyList = list[t.Any]

    PipelineGenerator = t.Generator[transformers_ext.TransformersPipeline, None, None]

framework = bentoml.transformers

backward_compatible = True

TINY_TEXT_MODEL = "hf-internal-testing/tiny-random-distilbert"
TINY_TEXT_TASK = get_task(TINY_TEXT_MODEL)

set_seed(124)


def softmax(outputs: NDArray[t.Any]) -> NDArray[t.Any]:
    maxes = np.max(outputs, axis=-1, keepdims=True)
    shifted_exp = np.exp(outputs - maxes)
    return shifted_exp / shifted_exp.sum(axis=-1, keepdims=True)


class PairClassificationPipeline(Pipeline):
    def _sanitize_parameters(self, **kwargs: t.Any) -> tuple[AnyDict, AnyDict, AnyDict]:
        preprocess_kwargs: AnyDict = {}
        if "second_text" in kwargs:
            preprocess_kwargs["second_text"] = kwargs["second_text"]
        return preprocess_kwargs, {}, {}

    def preprocess(self, f: str, second_text: str | None = None) -> BatchEncoding:
        assert self.tokenizer is not None
        return self.tokenizer(f, text_pair=second_text, return_tensors=self.framework)

    def _forward(self, input_tensors: dict[str, GenericTensor]) -> ModelOutput:
        return t.cast("ModelOutput", self.model(**input_tensors))

    def postprocess(self, model_outputs: ModelOutput) -> dict[str, t.Any]:
        assert self.model.config.id2label is not None
        logits: NDArray[t.Any] = t.cast("torch.Tensor", model_outputs.logits[0]).numpy()
        probabilities = softmax(logits)

        best_class = np.argmax(probabilities)
        label = t.cast(str, self.model.config.id2label[best_class])
        score = probabilities[best_class].item()
        logits = logits.tolist()
        return {"label": label, "score": score, "logits": logits}


def gen_task_pipeline(
    model: str, task: str | None = None, *, frameworks: list[str] = ["pt", "tf"]
) -> PipelineGenerator:
    yield from (
        transformers.pipeline(task=task, model=model, framework=f) for f in frameworks
    )


def expected_equal(
    expected: list[AnyDict | AnyList],
) -> t.Callable[[list[AnyDict | AnyList]], bool]:
    def check_output(out: list[AnyDict | AnyList]) -> bool:
        return nested_simplify(out, decimals=4) == expected

    return check_output


text_classification_pipeline: list[Model] = [
    Model(
        name="text-classification-pipeline",
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
    for model in gen_task_pipeline(model=TINY_TEXT_MODEL, task=TINY_TEXT_TASK)
]

batched_pipeline: list[Model] = [
    Model(
        name="batchable-text-classification-pipeline",
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
    for model in gen_task_pipeline(model=TINY_TEXT_MODEL, task=TINY_TEXT_TASK)
]

tiny_image_model = "hf-internal-testing/tiny-random-vit-gpt2"
tiny_image_task = "image-to-text"
test_url = "http://images.cocodataset.org/val2017/000000039769.jpg"

image_classification: list[Model] = [
    Model(
        name="image-to-text-pipeline",
        model=model,
        configurations=[
            Config(
                load_kwargs={"task": tiny_image_task},
                test_inputs={
                    "__call__": [
                        Input(
                            input_args=[[test_url]],
                            expected=[
                                [
                                    {
                                        "generated_text": "growthgrowthgrowthgrowthgrowthgrowthgrowthgrowthgrowthgrowthgrowthgrowthgrowthgrowthgrowthgrowthgrowthGOGO"
                                    },
                                ]
                            ],
                        ),
                        Input(
                            input_args=[
                                [Image.open(requests.get(test_url, stream=True).raw)]
                            ],
                            expected=[
                                [
                                    {
                                        "generated_text": "growthgrowthgrowthgrowthgrowthgrowthgrowthgrowthgrowthgrowthgrowthgrowthgrowthgrowthgrowthgrowthgrowthGOGO"
                                    },
                                ]
                            ],
                        ),
                    ],
                },
            ),
        ],
    )
    for model in gen_task_pipeline(model=tiny_image_model, task=tiny_image_task)
]


def gen_custom_pipeline_kwargs(
    task_name: str | None = None,
    /,
    *,
    pt_model_auto: str = "AutoModelForSequenceClassification",
    tf_model_auto: str = "TFAutoModelForSequenceClassification",
    model_name: str = TINY_TEXT_MODEL,
    task_type: str = "text",
) -> tuple[str, TaskDefinition]:
    if task_name is None:
        task_name = get_task(model_name)
    return task_name, TaskDefinition(
        impl=PairClassificationPipeline,
        tf=convert_to_autoclass(tf_model_auto) if is_tf_available() else None,
        pt=convert_to_autoclass(pt_model_auto) if is_torch_available() else None,
        default=SimpleDefaultMapping(pt=(model_name,)),
        type=task_type,
    )


custom_task, definition = gen_custom_pipeline_kwargs("custom-text-classification")

register_pipeline(custom_task, **definition)


def check_model(_: transformers_ext.TransformersPipeline, __: AnyDict) -> None:
    assert custom_task in get_supported_tasks()


# NOTE: Pipeline with Tensorflow does work when the custom pipelines
# are published on the HuggingFace Hub. Otherwise, it is not possible
# to pickle Tensorflow Model.
custom_pipeline: list[Model] = [
    Model(
        name="custom_text_classification_pipeline",
        model=model,
        save_kwargs={
            "task_name": custom_task,
            "task_definition": definition,
            "external_modules": [
                importlib.import_module(".", "tests.integration.frameworks.models")
            ],
        },
        configurations=[
            Config(
                load_kwargs={
                    "pt": definition["pt"],
                    "tf": definition["tf"],
                    "default": definition.get("default", {}),
                    "type": definition.get("type", "text"),
                },
                test_inputs={
                    "__call__": [
                        Input(
                            input_args=["i love you"],
                            expected=lambda out: nested_simplify(
                                out["score"], decimals=4
                            )
                            == 0.5036,
                        ),
                        Input(
                            input_args=["I hate you"],
                            input_kwargs={"second_text": "I love you"},
                            expected=lambda out: nested_simplify(
                                out["score"], decimals=4
                            )
                            == 0.5036,
                        ),
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

delete_pipeline(custom_task)

if t.TYPE_CHECKING:

    class FrameworkPreTrainedMapping(t.TypedDict):
        pt: str
        tf: str
        jax: str

else:
    FrameworkPreTrainedMapping = dict

# NOTE: the below is a map between model saved weights and its coresponding dictionary of framework - pretrained class.
# TODO: Add audio models
MODEL_PRETRAINED_MAPPING = {
    # NOTE: text model
    "gpt2": FrameworkPreTrainedMapping(
        pt="GPT2Model", tf="TFGPT2Model", jax="FlaxGPT2Model"
    ),
    # NOTE: vit model
    "google/vit-base-patch16-224-in21k": FrameworkPreTrainedMapping(
        pt="ViTModel", tf="TFViTModel", jax="FlaxViTModel"
    ),
}


def pretrained_model(
    framework: t.Literal["pt", "tf", "jax"], model: str, /
) -> (
    transformers.PreTrainedModel
    | transformers.FlaxPreTrainedModel
    | transformers.TFPreTrainedModel
):
    if framework in MODEL_PRETRAINED_MAPPING[model]:
        return getattr(
            transformers, MODEL_PRETRAINED_MAPPING[model][framework]
        ).from_pretrained(model)
    raise ValueError(
        f"Framework {framework} doesn't have a pretrained class implementation."
    )


def gpt2_method_caller(
    framework: t.Literal["pt", "tf", "jax"], model_name: str
) -> t.Callable[
    [Model, str, tuple[t.Any, ...], dict[str, t.Any]], tuple[int, int, int]
]:
    def caller(
        framework_test_model: Model,
        method: str,
        args: tuple[t.Any, ...],
        kwargs: dict[str, t.Any],
    ) -> t.Any:
        tokenizer = transformers.AutoTokenizer.from_pretrained(model_name)
        model_inputs = kwargs
        if args:
            model_inputs.update(
                t.cast("dict[str, t.Any]", tokenizer(args, return_tensors=framework))
            )
        last_hidden_state = getattr(framework_test_model.model, method)(
            **model_inputs
        ).last_hidden_state
        return last_hidden_state.shape

    return caller


def check_gpt2_output(output: tuple[int, int, int] | t.Any) -> bool:
    expected_shape = (1, 6, 768)
    if isinstance(output, tuple):
        return output == expected_shape
    elif isinstance(output, tf.TensorShape):  # eh a special case for tf.
        return output == tf.TensorShape(expected_shape)
    else:
        return output.last_hidden_state.shape == expected_shape


gpt2_tokenizer = transformers.AutoTokenizer.from_pretrained("gpt2")

gpt2_pretrained: list[Model] = [
    Model(
        name=f"gpt2_{framework}",
        model=pretrained_model(framework, "gpt2"),
        model_method_caller=gpt2_method_caller(framework, "gpt2"),
        configurations=[
            Config(
                test_inputs={
                    "__call__": [
                        Input(
                            input_args=[],
                            input_kwargs=gpt2_tokenizer(
                                "Hello, my dog is cute", return_tensors=framework
                            ),
                            expected=check_gpt2_output,
                        )
                    ]
                }
            )
        ],
    )
    for framework in ("pt", "tf", "jax")
] + [
    Model(
        name="gpt2_tokenizer",
        model=gpt2_tokenizer,
        configurations=[
            Config(
                test_inputs={
                    "__call__": [
                        Input(
                            input_args=["Hello, my dog is cute"],
                            expected={
                                "input_ids": [15496, 11, 616, 3290, 318, 13779],
                                "attention_mask": [1, 1, 1, 1, 1, 1],
                            },
                        )
                    ]
                }
            )
        ],
    ),
]

dataset = load_dataset("huggingface/cats-image")
im = dataset["test"]["image"][0]
vit_processor = transformers.AutoImageProcessor.from_pretrained(
    "google/vit-base-patch16-224-in21k"
)


def vit_method_caller(
    framework: t.Literal["pt", "tf", "jax", "np"], model_name: str
) -> t.Callable[
    [Model, str, tuple[t.Any, ...], dict[str, t.Any]], tuple[int, int, int]
]:
    if framework == "jax":
        # NOTE: for jax ViT it returns a numpy array instead.
        framework = "np"

    def caller(
        framework_test_model: Model,
        method: str,
        args: tuple[t.Any, ...],
        kwargs: dict[str, t.Any],
    ) -> t.Any:
        model_inputs = kwargs
        if args:
            model_inputs.update(
                t.cast(
                    "dict[str, t.Any]", vit_processor(args, return_tensors=framework)
                )
            )
        last_hidden_state = getattr(framework_test_model.model, method)(
            **model_inputs
        ).last_hidden_state
        return last_hidden_state.shape

    return caller


def check_vit_output(output: tuple[int, int, int] | t.Any) -> bool:
    expected_shape = (1, 197, 768)
    if isinstance(output, tuple):
        return output == expected_shape
    elif isinstance(output, tf.TensorShape):  # eh a special case for tf.
        return output == tf.TensorShape(expected_shape)
    else:
        return output.last_hidden_state.shape == expected_shape


vit_pretrained: list[Model] = [
    Model(
        name=f"vit_{framework}",
        model=pretrained_model(framework, "google/vit-base-patch16-224-in21k"),
        model_method_caller=vit_method_caller(
            framework, "google/vit-base-patch16-224-in21k"
        ),
        configurations=[
            Config(
                test_inputs={
                    "__call__": [
                        Input(
                            input_args=[],
                            input_kwargs=vit_processor(im, return_tensors=framework),
                            expected=check_vit_output,
                        )
                    ]
                }
            )
        ],
    )
    for framework in ("pt", "tf", "jax")
] + [
    Model(
        name="vit_processor",
        model=vit_processor,
        configurations=[
            Config(
                test_inputs={
                    "__call__": [
                        Input(
                            input_args=[im],
                            expected=lambda output: output["pixel_values"][0].shape
                            == (3, 224, 224),
                        )
                    ]
                }
            )
        ],
    ),
]


# NOTE: when we need to add more test cases for different models
#  create a list of Model and append to 'models' list
models = (
    text_classification_pipeline
    + batched_pipeline
    + image_classification
    + custom_pipeline
    + gpt2_pretrained
    + vit_pretrained
)
