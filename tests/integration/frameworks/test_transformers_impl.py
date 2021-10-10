import os
import typing as t

import pytest
from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed

import bentoml.transformers
from bentoml.exceptions import InvalidArgument, NotFound

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


@pytest.fixture(scope="session")
def gpt_model() -> t.Dict[str, t.Union[AutoTokenizer, AutoModelForCausalLM]]:
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    model = AutoModelForCausalLM.from_pretrained(
        "gpt2", pad_token_id=tokenizer.eos_token_id
    )
    return {"model": model, "tokenizer": tokenizer}


def test_transformers_save_load(tmpdir, gpt_model):
    ...
