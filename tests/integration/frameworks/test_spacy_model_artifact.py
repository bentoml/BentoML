import os
import random
import typing as t

import pytest
import spacy
from spacy.training import Example
from spacy.util import minibatch

from bentoml.spacy import SpacyModel

train_data: t.List[t.Tuple[str, dict]] = [
    ("Google has changed the logo of its apps", {"entities": [(0, 6, "ORG")]}),
    ("Facebook has introduced a new app!", {"entities": [(0, 8, "ORG")]}),
    ("Amazon has partnered with small businesses.", {"entities": [(0, 6, "ORG")]}),
]

test_json: t.Dict[str, str] = {"text": "Google cloud is now a separate entity."}


def predict_json(model: "spacy.language.Language", json: t.Dict[str, str]) -> str:
    return model(json["text"]).text


@pytest.fixture(scope="module")
def spacy_model():
    examples = []
    model = spacy.blank("en")
    if "ner" not in model.pipe_names:
        ner = model.add_pipe("ner", last=True)
    else:
        ner = model.get_pipe("ner")

    for text, annotations in train_data:
        examples.append(Example.from_dict(model.make_doc(text), annotations))
        for ent in annotations.get("entities"):
            ner.add_label(ent[2])

    other_pipes = [pipe for pipe in model.pipe_names if pipe != "ner"]

    with model.disable_pipes(*other_pipes):
        optimizer = model.begin_training()
        for _ in range(10):
            random.shuffle(examples)
            for batch in minibatch(examples, size=8):
                model.update(batch, sgd=optimizer)

    return model


@pytest.mark.parametrize(
    "loaded_pipe, predict_fn",
    [
        (SpacyModel.load("en_core_web_sm"), predict_json),
    ],
)
def test_spacy_save_load(tmpdir, spacy_model, loaded_pipe, predict_fn):
    SpacyModel(spacy_model).save(tmpdir)
    assert os.path.exists(os.path.join(tmpdir, "bentoml_model"))
    spacy_loaded: spacy.language.Language = SpacyModel.load(tmpdir)
    assert predict_fn(spacy_loaded, test_json) == test_json["text"]
    assert predict_fn(loaded_pipe, test_json) == test_json["text"]
