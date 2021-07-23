import os

import pytest
from transformers import AutoModelWithLMHead, AutoTokenizer

from bentoml.transformers import TransformersModel


def predict_json(gpt, jsons):
    text = jsons.get("text")
    model = gpt.get("model")
    tokenizer = gpt.get("tokenizer")
    input_ids = tokenizer.encode(text, return_tensors="pt")
    output = model.generate(input_ids, max_length=50)
    return tokenizer.decode(output[0], skip_special_tokens=True)


test_sentence = {"text": "A Bento box is a "}

result = (
    "A Bento box is a urn that is used to store the contents of a Bento box. "
    + "It is usually used to store the contents of a Bento box in a storage container."
    + "\n\nThe Bento box is a small, rectangular"
)


@pytest.fixture(scope="module")
def gpt_model() -> dict:
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    model = AutoModelWithLMHead.from_pretrained(
        "gpt2", pad_token_id=tokenizer.eos_token_id
    )
    return {"model": model, "tokenizer": tokenizer}


def test_transformers_save_load(tmpdir, gpt_model):
    TransformersModel(gpt_model).save(tmpdir)
    assert os.path.exists(os.path.join(tmpdir, "tokenizer_type.txt"))

    gpt2_loaded = TransformersModel.load(tmpdir)
    assert predict_json(gpt2_loaded, test_sentence) == predict_json(
        gpt_model, test_sentence
    )
