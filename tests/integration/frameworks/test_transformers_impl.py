import pytest
import requests
import transformers.pipelines
from transformers import set_seed
from transformers.file_utils import CONFIG_NAME, hf_bucket_url
from transformers.testing_utils import DUMMY_UNKWOWN_IDENTIFIER as MODEL_ID

import bentoml.transformers
from bentoml.exceptions import BentoMLException
from tests.utils.helpers import assert_have_file_extension

set_seed(123)

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


def generate_from_text(model, tokenizer, jsons, return_tensors="pt"):
    text = jsons.get("text")
    input_ids = tokenizer.encode(text, return_tensors=return_tensors)
    output = model.generate(
        input_ids, max_length=50, pad_token_id=tokenizer.eos_token_id
    )
    return tokenizer.decode(output[0], skip_special_tokens=True)


@pytest.mark.parametrize(
    "autoclass, exc",
    [
        ({"framework": "xgboost", "lm_head": "test"}, AttributeError),
        ({"framework": "xgb", "lm_head": "test"}, AttributeError),
        ({"framework": "pt", "lm_head": "test"}, AttributeError),
        ({"framework": "flax", "lm_head": "ctc"}, BentoMLException),
    ],
)
def test_load_autoclass(autoclass, exc):
    with pytest.raises(exc):
        bentoml.transformers._load_autoclass(**autoclass)


@pytest.mark.parametrize(
    "kwargs, exc",
    [
        (
            {"hf_url": "https://bofa", "output_dir": "/tmp"},
            requests.exceptions.ConnectionError,
        ),
        (
            {
                "hf_url": hf_bucket_url(MODEL_ID, filename="missing.bin"),
                "output_dir": "/tmp",
            },
            requests.exceptions.HTTPError,
        ),
        (
            {
                "hf_url": hf_bucket_url(
                    MODEL_ID, filename=CONFIG_NAME, revision=REVISION_ID_INVALID
                ),
                "output_dir": "/tmp",
            },
            requests.exceptions.HTTPError,
        ),
    ],
)
def test_download_from_hub(kwargs, exc):
    with pytest.raises(exc):
        bentoml.transformers._download_from_hub(**kwargs)


@pytest.mark.parametrize(
    "kwargs",
    [
        ({"from_tf": True, "keep_download_from_hub": True}),
        ({"from_flax": True, "keep_download_from_hub": True}),
    ],
)
def test_transformers_import_from_huggingface_hub(modelstore, kwargs):
    tag = bentoml.transformers.import_from_huggingface_hub(
        model_name, model_store=modelstore, **kwargs
    )
    info = modelstore.get(tag)
    try:
        if kwargs["from_tf"]:
            assert_have_file_extension(
                info.path,
                ".h5",
            )
    except KeyError:
        assert_have_file_extension(
            info.path,
            ".msgpack",
        )


@pytest.mark.parametrize(
    "kwargs, frameworks, tensors_type",
    [
        ({"from_tf": False, "keep_download_from_hub": True}, "pt", "pt"),
        ({"from_tf": True}, "tf", "tf"),
    ],
)
def test_transformers_save_load(modelstore, frameworks, tensors_type, kwargs):
    tag = bentoml.transformers.import_from_huggingface_hub(
        "gpt2", model_store=modelstore, **kwargs
    )
    _, model, tokenizer = bentoml.transformers.load(
        tag, framework=frameworks, model_store=modelstore
    )
    assert (
        generate_from_text(model, tokenizer, test_sentence, return_tensors=tensors_type)
        == result
    )


def test_transformers_runner_setup_run_batch(modelstore):
    tag = bentoml.transformers.import_from_huggingface_hub(
        "distilbert-base-uncased-finetuned-sst-2-english", model_store=modelstore
    )
    runner = bentoml.transformers.load_runner(
        tag, tasks="text-classification", model_store=modelstore
    )
    assert tag in runner.required_models
    assert runner.num_concurrency_per_replica == runner.num_replica == 1

    res = runner.run_batch(batched_sentence)
    assert all(i["score"] >= 0.8 for i in res)
    assert isinstance(runner._pipeline, transformers.pipelines.Pipeline)


def test_transformers_runner_pipelines_kwargs(modelstore):
    from PIL import Image

    tag = bentoml.transformers.import_from_huggingface_hub(
        "google/vit-large-patch16-224", model_store=modelstore
    )
    runner = bentoml.transformers.load_runner(
        tag,
        tasks="image-classification",
        device=-1,
        feature_extractor="google/vit-large-patch16-224",
        model_store=modelstore,
    )
    url = "http://images.cocodataset.org/val2017/000000039769.jpg"
    image = Image.open(requests.get(url, stream=True).raw)
    res = runner.run_batch(image)
    assert res[0]["label"] == "Egyptian cat"
