import os

import pytest
import requests
import transformers.models.gpt2.modeling_gpt2
from transformers import set_seed
from transformers.file_utils import CONFIG_NAME, hf_bucket_url
from transformers.testing_utils import DUMMY_UNKWOWN_IDENTIFIER as MODEL_ID

import bentoml.transformers
from bentoml.exceptions import BentoMLException
from tests._internal.helpers import assert_have_file_extension

REVISION_ID_INVALID = "aaaaaaa"

set_seed(123)

model_name = "gpt2"
test_sentence = {"text": "A Bento box is a "}
result = (
    "A Bento box is a urn that is used to store the contents of a Bento box. "
    + "It is usually used to store the contents of a Bento box in a storage container."
    + "\n\nThe Bento box is a small, rectangular"
)


def generate_from_text(model, tokenizer, jsons, return_tensors="pt"):
    text = jsons.get("text")
    input_ids = tokenizer.encode(text, return_tensors=return_tensors)
    output = model.generate(input_ids, max_length=50)
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
        ({"from_tf": True}),
        ({"from_flax": True}),
    ],
)
def test_transformers_import_from_huggingface_hub(modelstore, kwargs):
    tag = bentoml.transformers.import_from_huggingface_hub(
        model_name, model_store=modelstore, **kwargs
    )
    try:
        if kwargs["from_tf"]:
            assert_have_file_extension(
                os.path.join(modelstore._base_dir, model_name, tag.split(":")[-1]),
                ".h5",
            )
    except KeyError:
        assert_have_file_extension(
            os.path.join(modelstore._base_dir, model_name, tag.split(":")[-1]),
            ".msgpack",
        )


@pytest.mark.parametrize(
    "kwargs, frameworks, tensors_type",
    [({"from_tf": False}, "pt", "pt"), ({"from_tf": True}, "tf", "tf")],
)
def test_transformers_save_load(modelstore, frameworks, tensors_type, kwargs):
    tag = bentoml.transformers.import_from_huggingface_hub(
        "gpt2", model_store=modelstore, **kwargs
    )
    model, tokenizer = bentoml.transformers.load(
        tag, framework=frameworks, model_store=modelstore
    )
    assert (
        generate_from_text(model, tokenizer, test_sentence, return_tensors=tensors_type)
        == result
    )
