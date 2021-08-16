import os
import typing as t

import pytest
from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed

from bentoml.exceptions import InvalidArgument, NotFound
from bentoml.transformers import TransformersModel

set_seed(123)

test_sentence = {"text": "A Bento box is a "}

result = (
    "A Bento box is a urn that is used to store the contents of a Bento box. "
    + "It is usually used to store the contents of a Bento box in a storage container."
    + "\n\nThe Bento box is a small, rectangular"
)


def generate_from_text(gpt, jsons, return_tensors="pt"):
    text = jsons.get("text")
    model, tokenizer = gpt.values()
    input_ids = tokenizer.encode(text, return_tensors=return_tensors)
    output = model.generate(input_ids, max_length=50)
    return tokenizer.decode(output[0], skip_special_tokens=True)


def create_invalid_transformers_class(name):
    class Foo:
        pass

    Foo.__module__ = name
    return Foo


@pytest.fixture(scope="session")
def gpt_model() -> t.Dict[str, t.Union[AutoTokenizer, AutoModelForCausalLM]]:
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    model = AutoModelForCausalLM.from_pretrained(
        "gpt2", pad_token_id=tokenizer.eos_token_id
    )
    return {"model": model, "tokenizer": tokenizer}


def test_transformers_save_load(tmpdir, gpt_model):
    TransformersModel(gpt_model).save(tmpdir)
    assert os.path.exists(os.path.join(tmpdir, "__tokenizer_class_type.txt"))

    gpt2_loaded = TransformersModel.load(tmpdir)
    assert generate_from_text(gpt2_loaded, test_sentence) == result


def test_transformers_load_from_dict(gpt_model):
    loaded = TransformersModel.load(gpt_model)
    assert generate_from_text(loaded, test_sentence) == result


@pytest.mark.parametrize(
    "invalid_dict, exc",
    [
        ({"test": "doesn't have model"}, InvalidArgument),
        ({"model": "hi", "test": "doesn't have tokenizer"}, InvalidArgument),
        (
            {
                "model": create_invalid_transformers_class("foo"),
                "tokenizer": create_invalid_transformers_class("bar"),
            },
            InvalidArgument,
        ),
        (
            {
                "model": create_invalid_transformers_class("transformers"),
                "tokenizer": create_invalid_transformers_class("foo"),
            },
            InvalidArgument,
        ),
        ("FooBar", NotFound),
    ],
)
def test_invalid_transformers_load(invalid_dict, exc):
    with pytest.raises(exc):
        TransformersModel.load(invalid_dict)


@pytest.mark.parametrize("frameworks, tensors_type", [("pt", "pt"), ("tf", "tf")])
def test_transformers_load_frameworks(frameworks, tensors_type):
    loaded = TransformersModel.load("gpt2", framework=frameworks)
    assert (
        generate_from_text(loaded, test_sentence, return_tensors=tensors_type) == result
    )
