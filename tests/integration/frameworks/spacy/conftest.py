import random
import typing as t

import pytest
import spacy
from spacy.training import Example
from spacy.util import minibatch

train_data: t.List[t.Tuple[str, t.Dict[str, t.List[t.Tuple[int, int, str]]]]] = [
    ("Google has changed the logo of its apps", {"entities": [(0, 6, "ORG")]}),
    ("Facebook has introduced a new app!", {"entities": [(0, 8, "ORG")]}),
    ("Amazon has partnered with small businesses.", {"entities": [(0, 6, "ORG")]}),
]


@pytest.fixture(scope="module")
def spacy_model() -> spacy.language.Language:
    examples: t.List[t.Any] = []
    model = spacy.blank("en")
    if "ner" not in model.pipe_names:
        ner = model.add_pipe("ner", last=True)
    else:
        ner = model.get_pipe("ner")

    for text, annotations in train_data:
        examples.append(Example.from_dict(model.make_doc(text), annotations))  # noqa
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
