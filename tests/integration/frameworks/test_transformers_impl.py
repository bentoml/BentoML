import typing as t
from typing import TYPE_CHECKING

import pytest
import requests
import transformers.pipelines
from transformers.trainer_utils import set_seed

import bentoml

if TYPE_CHECKING:
    from bentoml._internal.external_typing import transformers as ext


set_seed(123)

MODEL_ID = "julien-c/dummy-unknown"

REVISION_ID_INVALID = "e10"

model_name = "gpt2"
test_sentence = {"text": "A Bento box is a "}
batched_sentence = [
    "I love you and I want to spend my whole life with you",
    "I hate you, Lyon, you broke my heart.",
]
result = (
    "A Bento box is a urn that is used to store the contents of a Bento box. "
    + "It is usually used to store the contents of a Bento box in a storage container."
    + "\n\nThe Bento box is a small, rectangular"
)


def generate_from_text(
    model: "ext.TransformersModelType",
    tokenizer: "ext.TransformersTokenizerType",
    jsons: t.Dict[str, str],
    return_tensors: str = "pt",
) -> str:
    text = jsons.get("text")
    input_ids = tokenizer.encode(text, return_tensors=return_tensors)
    output = model.generate(
        input_ids, max_length=50, pad_token_id=tokenizer.eos_token_id
    )
    return tokenizer.decode(output[0], skip_special_tokens=True)


@pytest.mark.parametrize(
    "kwargs, framework, tensors_type",
    [
        ({"from_tf": False}, "pt", "pt"),
        ({}, "tf", "tf"),
    ],
)
def test_transformers_save_load(
    framework: str,
    tensors_type: str,
    kwargs: t.Dict[str, t.Any],
):
    if "tf" in framework:
        loader = transformers.TFAutoModelForCausalLM
    else:
        loader = transformers.AutoModelForCausalLM
    model = loader.from_pretrained(model_name, **kwargs)
    tokenizer = transformers.AutoTokenizer.from_pretrained(model_name, **kwargs)
    tag = bentoml.transformers.save(model_name, model, tokenizer=tokenizer)
    lmodel, ltokenizer = bentoml.transformers.load(tag, from_tf="tf" in framework)
    res = generate_from_text(
        lmodel, ltokenizer, test_sentence, return_tensors=tensors_type
    )
    assert res == result


def test_transformers_save_load_pipeline():
    from PIL import Image

    pipeline = transformers.pipeline("image-classification")
    tag = bentoml.transformers.save("vit-image-classification", pipeline)
    url = "http://images.cocodataset.org/val2017/000000039769.jpg"
    image = Image.open(requests.get(url, stream=True).raw)
    model, fe = bentoml.transformers.load(tag)
    loaded = transformers.pipeline(
        "image-classification", model=model, feature_extractor=fe
    )
    res = loaded(image)
    assert res[0]["label"] == "Egyptian cat"


def test_transformers_runner_setup_run_batch():
    pipeline = transformers.pipeline("text-classification")
    tag = bentoml.transformers.save("text-classification-pipeline", pipeline)
    runner = bentoml.transformers.load_runner(tag, tasks="text-classification")
    assert tag in runner.required_models
    assert runner.num_replica == 1

    res = runner.run_batch(batched_sentence)
    assert all(i["score"] >= 0.4 for i in res)
    assert isinstance(runner._pipeline, transformers.pipelines.Pipeline)


def test_transformers_runner_pipelines_kwargs():
    from PIL import Image

    pipeline = transformers.pipeline("image-classification")
    tag = bentoml.transformers.save("vit-image-classification", pipeline)
    runner = bentoml.transformers.load_runner(tag, tasks="image-classification")
    url = "http://images.cocodataset.org/val2017/000000039769.jpg"
    image = Image.open(requests.get(url, stream=True).raw)
    res = runner.run_batch(image)
    assert res[0]["label"] == "Egyptian cat"
