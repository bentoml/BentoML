import typing as t
from typing import TYPE_CHECKING
from pathlib import Path

import yaml
import spacy
import pytest

import bentoml
from bentoml.exceptions import BentoMLException
from bentoml.exceptions import MissingDependencyException

current_file = Path(__file__).parent

MODEL_NAME = __name__.split(".")[-1]

test_json: t.Dict[str, str] = {"text": "Google cloud is now a separate entity."}


def predict_json(model: "spacy.language.Language", json: t.Dict[str, str]) -> str:
    return model(json["text"]).text


def test_spacy_save_load(spacy_model: "spacy.language.Language"):
    tag = bentoml.spacy.save(MODEL_NAME, spacy_model)
    model_info = bentoml.models.get(tag)
    assert "meta.json" in [_.name for _ in Path(model_info.path).iterdir()]

    spacy_loaded = bentoml.spacy.load(tag)
    assert predict_json(spacy_loaded, test_json) == test_json["text"]
    with pytest.raises(BentoMLException):
        _ = bentoml.spacy.load_project(tag)


def test_spacy_load_project():
    tag = bentoml.spacy.projects(
        "test",
        "clone",
        name="integrations/huggingface_hub",
        repo_or_store="https://github.com/aarnphm/projects",
    )
    path = bentoml.spacy.load_project(tag)
    assert "project.yml" in [f.name for f in Path(path).iterdir()]
    with pytest.raises(BentoMLException):
        _ = bentoml.spacy.load(tag)


def test_spacy_load_missing_deps_exc():
    nlp = spacy.load("en_core_web_sm")
    tag = bentoml.spacy.save("test_spacy", nlp)
    info = bentoml.models.get(tag)
    parent = info.path
    with Path(parent, "model.yaml").open("r") as f:
        content = yaml.safe_load(f)
    content["options"]["additional_requirements"] = ["spacy-transformers>=1.0.3,<1.1.0"]
    with Path(parent, "model.yaml").open("w") as of:
        yaml.dump(content, of)
    with pytest.raises(MissingDependencyException):
        _ = bentoml.spacy.load(tag)
