from pathlib import Path

import pytest
import spacy
import spacy.cli

import bentoml.spacy
from bentoml.exceptions import BentoMLException


def test_spacy_projects_clone(modelstore):
    tag, project_path = bentoml.spacy.projects(
        "test_spacy_project",
        "clone",
        name="integrations/huggingface_hub",
        repo_or_store="https://github.com/aarnphm/bentoml-spacy-projects-integration-tests",
        model_store=modelstore,
    )
    model_info = modelstore.get(tag)
    assert "project.yml" in [
        i.name for i in Path(model_info.path, "saved_model").iterdir()
    ]
    spacy.cli.project_assets(project_path)
    spacy.cli.project_run(project_path, "install")


def test_spacy_projects_pull(modelstore):
    project_yml = {
        "remotes": {
            "default": "https://github.com/aarnphm/bentoml-spacy-projects-integration-tests/tree/v3/pipelines/tagger_parser_ud",
        }
    }
    tag, project_path = bentoml.spacy.projects(
        "test_pull", "pull", remotes_config=project_yml, model_store=modelstore
    )
    model_info = modelstore.get(tag)
    assert "project.yml" in [
        i.name for i in Path(model_info.path, "saved_model").iterdir()
    ]


@pytest.mark.parametrize("tasks", ["assets", "document", "dvc", "push", "run"])
def test_spacy_projects_tasks_exc(modelstore, tasks):
    with pytest.raises(BentoMLException):
        _, _ = bentoml.spacy.projects("test", tasks, model_store=modelstore)
