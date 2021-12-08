from typing import TYPE_CHECKING
from pathlib import Path

import spacy
import pytest
import spacy.cli

import bentoml
from bentoml.exceptions import BentoMLException

if TYPE_CHECKING:
    from bentoml._internal.models import ModelStore


def test_spacy_projects_clone(modelstore: "ModelStore"):
    tag = bentoml.spacy.projects(
        "test_spacy_project",
        "clone",
        name="integrations/huggingface_hub",
        repo_or_store="https://github.com/aarnphm/bentoml-spacy-projects-integration-tests",
        model_store=modelstore,
    )
    project_path = bentoml.spacy.load_project(tag, model_store=modelstore)
    assert isinstance(project_path, str)
    assert "project.yml" in [i.name for i in Path(project_path).iterdir()]
    spacy.cli.project_assets(Path(project_path))
    spacy.cli.project_run(Path(project_path), "install")


def test_spacy_projects_pull(modelstore: "ModelStore"):
    project_yml = {
        "remotes": {
            "default": "https://github.com/aarnphm/bentoml-spacy-projects-integration-tests/tree/v3/pipelines/tagger_parser_ud",
        }
    }
    tag = bentoml.spacy.projects(
        "test_pull", "pull", remotes_config=project_yml, model_store=modelstore
    )
    project_path = bentoml.spacy.load_project(tag, model_store=modelstore)
    assert "project.yml" in [i.name for i in Path(project_path).iterdir()]


@pytest.mark.parametrize("tasks", ["assets", "document", "dvc", "push", "run"])
def test_spacy_projects_tasks_exc(modelstore: "ModelStore", tasks: "str"):
    with pytest.raises(BentoMLException):
        _ = bentoml.spacy.projects("test", tasks, model_store=modelstore)
