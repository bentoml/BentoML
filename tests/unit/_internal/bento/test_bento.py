import os
from typing import TYPE_CHECKING
from datetime import datetime
from datetime import timezone

import fs
import pytest

from bentoml._internal.bento import Bento
from bentoml._internal.types import Tag
from bentoml._internal.bento.bento import BentoInfo
from bentoml._internal.configuration import BENTOML_VERSION
from bentoml._internal.bento.build_config import BentoBuildConfig

if TYPE_CHECKING:
    from pathlib import Path

    from bentoml._internal.models import ModelStore


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

    with open(bento_yaml_b_filename, encoding="utf-8") as bento_yaml_b:
        assert bento_yaml_b.read() == expected_yaml.format(
            bentoml_version=BENTOML_VERSION,
            creation_time=bentoinfo_b.creation_time.isoformat(" "),
        )

    with open(bento_yaml_b_filename, encoding="utf-8") as bento_yaml_b:
        bentoinfo_b_from_yaml = BentoInfo.from_yaml_file(bento_yaml_b)

        assert bentoinfo_b_from_yaml == bentoinfo_b


def build_test_bento(model_store: "ModelStore") -> Bento:
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

    return Bento.create(
        bento_cfg,
        version="1.0",
        build_ctx="./simplebento",
        model_store=model_store,
    )


def fs_identical(fs1: fs.base.FS, fs2: fs.base.FS):
    for path in fs1.walk.dirs():
        assert fs2.isdir(path)

    for path in fs1.walk.files():
        assert fs2.isfile(path)
        assert fs1.readbytes(path) == fs2.readbytes(path)


@pytest.mark.usefixtures("change_test_dir")
def test_bento_export(tmpdir: "Path", dummy_model_store: "ModelStore"):
    working_dir = os.getcwd()

    testbento = build_test_bento(dummy_model_store)

    os.chdir(working_dir)

    cfg = BentoBuildConfig("bentoa.py:svc")
    bentoa = Bento.create(cfg, build_ctx="./bentoa")

    os.chdir(working_dir)

    bentoa1 = Bento.create(cfg, build_ctx="./bentoa1")

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
def test_bento(dummy_model_store: "ModelStore"):
    start = datetime.now(timezone.utc)
    bento = build_test_bento(dummy_model_store)
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
