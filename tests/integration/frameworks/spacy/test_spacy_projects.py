import pytest
import spacy

from pathlib import Path
import bentoml.spacy
from bentoml.exceptions import BentoMLException


def test_spacy_projects_clone(modelstore):
    tag, project_path = bentoml.spacy.projects(
        "clone",
        name="integrations/huggingface_hub",
        repo_or_store="https://github.com/aarnphm/projects",
        model_store=modelstore,
    )
    model_info = modelstore.get(tag)
    assert "project.yml" in [
        i.name for i in Path(model_info.path, "saved_model").iterdir()
    ]
    spacy.cli.project_assets(project_path)
    spacy.cli.project_run(project_path, "install")


def test_spacy_projects_pull(modelstore):
    ...


@pytest.mark.parametrize("tasks", ["assets", "document", "dvc", "push", "run"])
def test_spacy_projects_tasks_exc(modelstore, tasks):
    with pytest.raises(BentoMLException):
        _, _ = bentoml.spacy.projects(tasks, model_store=modelstore)
