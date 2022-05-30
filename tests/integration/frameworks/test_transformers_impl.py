import typing as t
from typing import TYPE_CHECKING

import pytest
import requests
import transformers
import transformers.pipelines
from PIL import Image
from transformers.trainer_utils import set_seed

import bentoml

if TYPE_CHECKING:
    from bentoml._internal.external_typing import transformers as ext


set_seed(124)


def tf_gpt2_pipeline():
    model = transformers.TFAutoModelForCausalLM.from_pretrained("gpt2")
    tokenizer = transformers.AutoTokenizer.from_pretrained("gpt2")
    return transformers.pipeline(
        task="text-generation", model=model, tokenizer=tokenizer
    )


def pt_gpt2_pipeline():
    model = transformers.AutoModelForCausalLM.from_pretrained("gpt2", from_tf=False)
    tokenizer = transformers.AutoTokenizer.from_pretrained("gpt2", from_tf=False)
    return transformers.pipeline(
        task="text-generation", model=model, tokenizer=tokenizer
    )


@pytest.mark.parametrize(
    "name, pipeline, with_options, expected_options, input_data",
    [
        (
            "text-generation",
            transformers.pipeline(task="text-generation"),  # type: ignore
            {},
            {"task": "text-generation", "kwargs": {}},
            "A Bento box is a ",
        ),
        (
            "text-generation",
            transformers.pipeline(task="text-generation"),  # type: ignore
            {"kwargs": {"a": 1}},
            {"task": "text-generation", "kwargs": {"a": 1}},
            "A Bento box is a ",
        ),
        (
            "text-generation",
            tf_gpt2_pipeline(),
            {},
            {"task": "text-generation", "kwargs": {}},
            "A Bento box is a ",
        ),
        (
            "text-generation",
            pt_gpt2_pipeline(),
            {},
            {"task": "text-generation", "kwargs": {}},
            "A Bento box is a ",
        ),
        (
            "image-classification",
            transformers.pipeline("image-classification"),  # type: ignore
            {},
            {"task": "image-classification", "kwargs": {}},
            Image.open(
                requests.get(
                    "http://images.cocodataset.org/val2017/000000039769.jpg",
                    stream=True,
                ).raw
            ),
        ),
        (
            "text-classification",
            transformers.pipeline("text-classification"),  # type: ignore
            {},
            {"task": "text-classification", "kwargs": {}},
            "BentoML is an awesome library for machine learning.",
        ),
    ],
)
def test_transformers(
    name: str,
    pipeline: "ext.TransformersPipelineType",  # type: ignore
    with_options: t.Dict[str, t.Any],
    expected_options: t.Dict[str, t.Any],
    input_data: t.Any,
):
    tag: bentoml.Tag = bentoml.transformers.save_model(name, pipeline)
    assert tag is not None
    assert tag.name == name

    bento_model: bentoml.Model = bentoml.transformers.get(tag).with_options(
        **with_options
    )
    assert bento_model.tag == tag
    assert bento_model.info.context.framework_name == "transformers"
    assert bento_model.info.options.task == expected_options["task"]  # type: ignore
    assert bento_model.info.options.kwargs == expected_options["kwargs"]  # type: ignore

    runnable: bentoml.Runnable = bentoml.transformers.get_runnable(bento_model)()
    output_data = runnable(input_data)  # type: ignore
    assert output_data is not None
