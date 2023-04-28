# pylint: disable=unused-argument
from __future__ import annotations

import os
from sys import version_info
from typing import TYPE_CHECKING
from datetime import datetime
from datetime import timezone

import fs
import pytest

from bentoml import Tag
from bentoml._internal.bento import Bento
from bentoml._internal.models import ModelStore
from bentoml._internal.bento.bento import BentoInfo
from bentoml._internal.bento.bento import BentoApiInfo
from bentoml._internal.bento.bento import BentoModelInfo
from bentoml._internal.bento.bento import BentoRunnerInfo
from bentoml._internal.configuration import BENTOML_VERSION
from bentoml._internal.bento.build_config import BentoBuildConfig

if TYPE_CHECKING:
    from pathlib import Path


def test_bento_info(tmpdir: Path):
    start = datetime.now(timezone.utc)
    bentoinfo_a = BentoInfo(tag=Tag("tag"), service="service")
    end = datetime.now(timezone.utc)

    assert bentoinfo_a.bentoml_version == BENTOML_VERSION
    assert start <= bentoinfo_a.creation_time <= end
    # validate should fail

    tag = Tag("test", "version")
    service = "testservice"
    labels = {"label": "stringvalue"}
    model_creation_time = datetime.now(timezone.utc)
    model_a = BentoModelInfo(
        tag=Tag("model_a", "v1"),
        module="model_a_module",
        creation_time=model_creation_time,
    )
    model_b = BentoModelInfo(
        tag=Tag("model_b", "v3"),
        module="model_b_module",
        creation_time=model_creation_time,
    )
    models = [model_a, model_b]
    runner_a = BentoRunnerInfo(
        name="runner_a",
        runnable_type="test_runnable_a",
        models=["runner_a_model"],
        resource_config={"cpu": 2},
    )
    runners = [runner_a]
    api_predict = BentoApiInfo(
        name="predict",
        input_type="NumpyNdarray",
        output_type="NumpyNdarray",
    )
    apis = [api_predict]

    bentoinfo_b = BentoInfo(
        tag=tag,
        service=service,
        labels=labels,
        runners=runners,
        models=models,
        apis=apis,
    )

    bento_yaml_b_filename = os.path.join(tmpdir, "b_dump.yml")
    with open(bento_yaml_b_filename, "w", encoding="utf-8") as bento_yaml_b:
        bentoinfo_b.dump(bento_yaml_b)

    expected_yaml = """\
service: testservice
name: test
version: version
bentoml_version: {bentoml_version}
creation_time: '{creation_time}'
labels:
  label: stringvalue
models:
- tag: model_a:v1
  module: model_a_module
  creation_time: '{model_creation_time}'
- tag: model_b:v3
  module: model_b_module
  creation_time: '{model_creation_time}'
runners:
- name: runner_a
  runnable_type: test_runnable_a
  embedded: false
  models:
  - runner_a_model
  resource_config:
    cpu: 2
apis:
- name: predict
  input_type: NumpyNdarray
  output_type: NumpyNdarray
docker:
  distro: debian
  python_version: '{python_version}'
  cuda_version: null
  env: null
  system_packages: null
  setup_script: null
  base_image: null
  dockerfile_template: null
python:
  requirements_txt: null
  packages: null
  lock_packages: true
  index_url: null
  no_index: null
  trusted_host: null
  find_links: null
  extra_index_url: null
  pip_args: null
  wheels: null
conda:
  environment_yml: null
  channels: null
  dependencies: null
  pip: null
"""

    with open(bento_yaml_b_filename, encoding="utf-8") as bento_yaml_b:
        assert bento_yaml_b.read() == expected_yaml.format(
            bentoml_version=BENTOML_VERSION,
            creation_time=bentoinfo_b.creation_time.isoformat(),
            model_creation_time=model_creation_time.isoformat(),
            python_version=f"{version_info.major}.{version_info.minor}",
        )

    with open(bento_yaml_b_filename, encoding="utf-8") as bento_yaml_b:
        bentoinfo_b_from_yaml = BentoInfo.from_yaml_file(bento_yaml_b)

        assert bentoinfo_b_from_yaml == bentoinfo_b


def build_test_bento() -> Bento:
    bento_cfg = BentoBuildConfig(
        "simplebento.py:svc",
        include=["*.py", "config.json", "somefile", "*dir*", ".bentoignore"],
        exclude=["*.storage", "/somefile", "/subdir2"],
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

    return Bento.create(bento_cfg, version="1.0", build_ctx="./simplebento")


def fs_identical(fs1: fs.base.FS, fs2: fs.base.FS):
    for path in fs1.walk.dirs():
        assert fs2.isdir(path)

    for path in fs1.walk.files():
        assert fs2.isfile(path)
        assert fs1.readbytes(path) == fs2.readbytes(path)


@pytest.mark.usefixtures("change_test_dir")
def test_bento_export(tmpdir: "Path", model_store: "ModelStore"):
    working_dir = os.getcwd()

    testbento = build_test_bento()
    # Bento build will change working dir to the build_context, this will reset it
    os.chdir(working_dir)

    cfg = BentoBuildConfig("bentoa.py:svc")
    bentoa = Bento.create(cfg, build_ctx="./bentoa")
    # Bento build will change working dir to the build_context, this will reset it
    os.chdir(working_dir)

    bentoa1 = Bento.create(cfg, build_ctx="./bentoa1")
    # Bento build will change working dir to the build_context, this will reset it
    os.chdir(working_dir)

    cfg = BentoBuildConfig("bentob.py:svc")
    bentob = Bento.create(cfg, build_ctx="./bentob")

    bento = testbento
    path = os.path.join(tmpdir, "testbento")
    export_path = bento.export(path)
    assert export_path == path + ".bento"
    assert os.path.isfile(export_path)
    imported_bento = Bento.import_from(export_path)
    assert imported_bento.tag == bento.tag
    assert imported_bento.info == bento.info
    del imported_bento

    bento = bentoa
    path = os.path.join(tmpdir, "bentoa")
    export_path = bento.export(path)
    assert export_path == path + ".bento"
    assert os.path.isfile(export_path)
    imported_bento = Bento.import_from(export_path)
    assert imported_bento.tag == bento.tag
    assert imported_bento.info == bento.info
    del imported_bento

    bento = bentoa1
    path = os.path.join(tmpdir, "bentoa1")
    export_path = bento.export(path)
    assert export_path == path + ".bento"
    assert os.path.isfile(export_path)
    imported_bento = Bento.import_from(export_path)
    assert imported_bento.tag == bento.tag
    assert imported_bento.info == bento.info
    del imported_bento

    bento = bentob
    path = os.path.join(tmpdir, "bentob")
    export_path = bento.export(path)
    assert export_path == path + ".bento"
    assert os.path.isfile(export_path)
    imported_bento = Bento.import_from(export_path)
    assert imported_bento.tag == bento.tag
    assert imported_bento.info == bento.info
    del imported_bento

    bento = testbento
    path = os.path.join(tmpdir, "testbento.bento")
    export_path = bento.export(path)
    assert export_path == path
    assert os.path.isfile(export_path)
    imported_bento = Bento.import_from(path)
    assert imported_bento.tag == bento.tag
    assert imported_bento.info == bento.info
    del imported_bento

    path = os.path.join(tmpdir, "testbento-parent")
    os.mkdir(path)
    export_path = bento.export(path)
    assert export_path == os.path.join(path, bento._export_name + ".bento")
    assert os.path.isfile(export_path)
    imported_bento = Bento.import_from(export_path)
    assert imported_bento.tag == bento.tag
    assert imported_bento.info == bento.info
    del imported_bento

    path = os.path.join(tmpdir, "testbento-parent-2/")
    with pytest.raises(ValueError):
        export_path = bento.export(path)

    path = os.path.join(tmpdir, "bento-dir")
    os.mkdir(path)
    export_path = bento.export(path)
    assert export_path == os.path.join(path, bento._export_name + ".bento")
    assert os.path.isfile(export_path)
    imported_bento = Bento.import_from(export_path)
    assert imported_bento.tag == bento.tag
    assert imported_bento.info == bento.info
    del imported_bento

    path = "temp://pytest-some-temp"
    export_path = bento.export(path)
    assert export_path.endswith(
        os.path.join("pytest-some-temp", bento._export_name + ".bento")
    )
    # because this is a tempdir, it's cleaned up immediately after creation...

    path = "osfs://" + fs.path.join(str(tmpdir), "testbento-by-url")
    export_path = bento.export(path)
    assert export_path == os.path.join(tmpdir, "testbento-by-url.bento")
    assert os.path.isfile(export_path)
    imported_bento = Bento.import_from(export_path)
    assert imported_bento.tag == bento.tag
    assert imported_bento.info == bento.info
    del imported_bento
    imported_bento = Bento.import_from(path + ".bento")
    assert imported_bento.tag == bento.tag
    assert imported_bento.info == bento.info
    del imported_bento

    path = "osfs://" + fs.path.join(str(tmpdir), "testbento-by-url")
    with pytest.raises(ValueError):
        bento.export(path, subpath="/badsubpath")

    path = "zip://" + fs.path.join(str(tmpdir), "testbento.zip")
    export_path = bento.export(path)
    assert export_path == path
    imported_bento = Bento.import_from(export_path)
    assert imported_bento.tag == bento.tag
    assert imported_bento.info == bento.info
    del imported_bento

    path = os.path.join(tmpdir, "testbento-gz")
    os.mkdir(path)
    export_path = bento.export(path, output_format="gz")
    assert export_path == os.path.join(path, bento._export_name + ".gz")
    assert os.path.isfile(export_path)
    imported_bento = Bento.import_from(export_path)
    assert imported_bento.tag == bento.tag
    assert imported_bento.info == bento.info
    del imported_bento

    path = os.path.join(tmpdir, "testbento-gz-1/")
    with pytest.raises(ValueError):
        bento.export(path, output_format="gz")


@pytest.mark.usefixtures("change_test_dir")
def test_bento(model_store: ModelStore):
    start = datetime.now(timezone.utc)
    bento = build_test_bento()
    end = datetime.now(timezone.utc)

    assert bento.info.bentoml_version == BENTOML_VERSION
    assert start <= bento.creation_time <= end
    # validate should fail

    with bento._fs as bento_fs:  # type: ignore
        assert set(bento_fs.listdir("/")) == {
            "bento.yaml",
            "apis",
            "models",
            "README.md",
            "src",
            "env",
        }
        assert set(bento_fs.listdir("src")) == {
            "simplebento.py",
            "subdir",
            ".bentoignore",
        }
        assert set(bento_fs.listdir("src/subdir")) == {"somefile"}
