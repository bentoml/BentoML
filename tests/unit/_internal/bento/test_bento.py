import os
from typing import TYPE_CHECKING
from datetime import datetime
from datetime import timezone

import pytest

from bentoml._internal.bento import Bento
from bentoml._internal.types import Tag
from bentoml._internal.bento.bento import BentoInfo
from bentoml._internal.configuration import BENTOML_VERSION
from bentoml._internal.bento.build_config import BentoBuildConfig

if TYPE_CHECKING:
    from pathlib import Path

    from bentoml._internal.models import ModelStore


expected_yaml = """\
service: testservice
name: test
version: version
bentoml_version: {bentoml_version}
creation_time: {creation_time}
labels:
  label: stringvalue
models:
- model:v1
- model2:latest
"""


def test_bento_info(tmpdir: "Path"):
    start = datetime.now(timezone.utc)
    bentoinfo_a = BentoInfo(Tag("tag"), "service", {}, [])
    end = datetime.now(timezone.utc)

    assert bentoinfo_a.bentoml_version == BENTOML_VERSION
    assert start <= bentoinfo_a.creation_time <= end
    # validate should fail

    tag = Tag("test", "version")
    service = "testservice"
    labels = {"label": "stringvalue"}
    models = [Tag("model", "v1"), Tag("model2", "latest")]
    bentoinfo_b = BentoInfo(tag, service, labels, models)

    bento_yaml_b_filename = os.path.join(tmpdir, "b_dump.yml")
    with open(bento_yaml_b_filename, "w", encoding="utf-8") as bento_yaml_b:
        bentoinfo_b.dump(bento_yaml_b)

    with open(bento_yaml_b_filename, encoding="utf-8") as bento_yaml_b:
        assert bento_yaml_b.read() == expected_yaml.format(
            bentoml_version=BENTOML_VERSION,
            creation_time=bentoinfo_b.creation_time.isoformat(" "),
        )

    with open(bento_yaml_b_filename, encoding="utf-8") as bento_yaml_b:
        bentoinfo_b_from_yaml = BentoInfo.from_yaml_file(bento_yaml_b)

        assert bentoinfo_b_from_yaml == bentoinfo_b


@pytest.mark.usefixtures("change_test_dir")
def test_bento(tmpdir: "Path", dummy_model_store: "ModelStore"):
    storepath = os.path.join(tmpdir, "bentos")
    os.makedirs(storepath, exist_ok=True)

    bento_cfg = BentoBuildConfig(
        "simplebento.py:svc",
        additional_models=[],
        include=["*.py", "config.json", "somefile", "*dir*", ".bentoignore"],
        exclude=[
            "*.storage",
            "/somefile",
        ],
        conda={
            "environment_yml": "./environment.yaml",
        },
        docker={
            "setup_script": "./setup_docker_container.sh",
        },
        labels={
            "team": "foo",
            "dataset_version": "abc",
            "framework": "pytorch",
        },
    )

    start = datetime.now(timezone.utc)
    bento = Bento.create(
        bento_cfg,
        version="1.0",
        build_ctx="./simplebento",
        model_store=dummy_model_store,
    )
    end = datetime.now(timezone.utc)

    assert bento.info.bentoml_version == BENTOML_VERSION
    assert start <= bento.creation_time <= end
    # validate should fail

    with bento._fs as fs:  # type: ignore
        assert set(fs.listdir("/")) == {
            "bento.yaml",
            "apis",
            "models",
            "README.md",
            "src",
            "env",
        }
        assert set(fs.listdir("src")) == {"simplebento.py", "subdir", ".bentoignore"}
        assert set(fs.listdir("src/subdir")) == {"somefile"}
