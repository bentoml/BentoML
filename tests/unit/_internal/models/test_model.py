from __future__ import annotations

import os
from sys import version_info as pyver
from typing import TYPE_CHECKING
from datetime import datetime
from datetime import timezone

import fs
import attr
import numpy as np
import pytest
import fs.errors

from bentoml import Tag
from bentoml.exceptions import BentoMLException
from bentoml.testing.pytest import TEST_MODEL_CONTEXT
from bentoml._internal.models import ModelOptions as InternalModelOptions
from bentoml._internal.models.model import Model
from bentoml._internal.models.model import ModelInfo
from bentoml._internal.models.model import ModelStore
from bentoml._internal.configuration import BENTOML_VERSION

if TYPE_CHECKING:
    from pathlib import Path

TEST_PYTHON_VERSION = f"{pyver.major}.{pyver.minor}.{pyver.micro}"

expected_yaml = """\
name: test
version: v1
module: test_model
labels:
  label: stringvalue
options:
  option_a: 1
  option_b: foo
  option_c:
  - 0.1
  - 0.2
metadata:
  a: 0.1
  b: 1
  c:
  - 2
  - 3
  - 4
context:
  framework_name: testing
  framework_versions:
    testing: v1
  bentoml_version: {bentoml_version}
  python_version: {python_version}
signatures:
  predict:
    batchable: false
  classify:
    batchable: true
    batch_dim:
    - 0
    - 0
  predict_ii:
    batchable: true
    batch_dim:
    - 0
    - 3
  classify_ii:
    batchable: true
    batch_dim:
    - 1
    - 3
api_version: v1
creation_time: '{creation_time}'
"""


@attr.define
class ModelOptions(InternalModelOptions):
    option_a: int
    option_b: str
    option_c: list[float]


@pytest.mark.usefixtures("change_test_dir")
def test_model_info(tmpdir: "Path"):
    start = datetime.now(timezone.utc)
    modelinfo_a = ModelInfo(
        tag=Tag("tag"),
        module="module",
        api_version="v1",
        labels={},
        options=ModelOptions(option_a=42, option_b="foo", option_c=[0.1, 0.2]),
        metadata={},
        context=TEST_MODEL_CONTEXT,
        signatures={"predict": {"batchable": True}},
    )
    end = datetime.now(timezone.utc)

    assert modelinfo_a.context.bentoml_version == BENTOML_VERSION
    assert modelinfo_a.context.python_version == TEST_PYTHON_VERSION
    assert start <= modelinfo_a.creation_time <= end

    tag = Tag("test", "v1")
    module = __name__
    labels = {"label": "stringvalue"}
    options = ModelOptions(option_a=1, option_b="foo", option_c=[0.1, 0.2])
    metadata = {"a": 0.1, "b": 1, "c": np.array([2, 3, 4], dtype=np.uint32)}
    # TODO: add test cases for input_spec and output_spec
    signatures = {
        "predict": {"batchable": False},
        "classify": {"batchable": True, "batch_dim": (0, 0)},
        "predict_ii": {"batchable": True, "batch_dim": (0, 3)},
        "classify_ii": {"batchable": True, "batch_dim": (1, 3)},
    }

    modelinfo_b = ModelInfo(
        tag=tag,
        module=module,
        api_version="v1",
        labels=labels,
        options=options,
        metadata=metadata,
        context=TEST_MODEL_CONTEXT,
        signatures=signatures,
    )

    model_yaml_b_filename = os.path.join(tmpdir, "b_dump.yml")
    with open(model_yaml_b_filename, "w", encoding="utf-8") as model_yaml_b:
        modelinfo_b.dump(model_yaml_b)

    with open(model_yaml_b_filename, encoding="utf-8") as model_yaml_b:
        assert model_yaml_b.read() == expected_yaml.format(
            bentoml_version=BENTOML_VERSION,
            creation_time=modelinfo_b.creation_time.isoformat(),
            python_version=TEST_PYTHON_VERSION,
        )

    with open(model_yaml_b_filename, encoding="utf-8") as model_yaml_b:
        modelinfo_b_from_yaml = ModelInfo.from_yaml_file(model_yaml_b)
        assert modelinfo_b_from_yaml == modelinfo_b

    # attempt to test that serialization is deterministic
    det_check_filename = os.path.join(tmpdir, "det_check.yml")
    with open(det_check_filename, "a+", encoding="utf-8") as det_check_yaml:
        modelinfo_b.dump(det_check_yaml)
        old_info = det_check_yaml.read()

        # re-flush
        modelinfo_b.dump(det_check_yaml)
        assert det_check_yaml.read() == old_info


def test_model_creationtime():
    start = datetime.now(timezone.utc)
    model_a = Model.create(
        "testmodel",
        module="test",
        api_version="v1",
        signatures={},
        context=TEST_MODEL_CONTEXT,
    )
    end = datetime.now(timezone.utc)

    assert model_a.tag.name == "testmodel"
    assert start <= model_a.creation_time <= end
    assert str(model_a) == f'Model(tag="{model_a.tag}")'
    assert repr(model_a) == f'Model(tag="{model_a.tag}", path="{model_a.path}")'


def test_model_version():
    model_with_version = Model.create(
        "testmodel:myversion",
        module="test",
        api_version="v1",
        signatures={},
        context=TEST_MODEL_CONTEXT,
    )

    assert model_with_version.info.version == "myversion"


class AdditionClass:
    def __init__(self, x: int):
        self.x = x

    def __call__(self, y: int) -> int:
        return self.x + y


add_num_1 = 5


@pytest.fixture(name="bento_model")
def fixture_bento_model():
    model = Model.create(
        "testmodel",
        module=__name__,
        api_version="v1",
        signatures={},
        context=TEST_MODEL_CONTEXT,
        options=ModelOptions(option_a=1, option_b="foo", option_c=[0.1, 0.2]),
        custom_objects={
            "add": AdditionClass(add_num_1),
        },
    )
    model.flush()
    return model


def test_model_equal(bento_model):
    # note: models are currently considered to be equal if their tag is equal;
    #       this is a test of that behavior
    eq_to_b = Model.create(
        "tmp", module="bar", api_version="v1", signatures={}, context=TEST_MODEL_CONTEXT
    )
    eq_to_b._tag = bento_model._tag  # type: ignore

    assert eq_to_b == bento_model
    assert eq_to_b.__hash__() == bento_model.__hash__()


def test_model_export_import(bento_model, tmpdir: "Path"):
    # note: these tests rely on created models having a system path
    sys_written_path = bento_model.path_of("sys_written/file")
    assert sys_written_path == os.path.join(bento_model.path, "sys_written", "file")

    os.makedirs(os.path.dirname(sys_written_path))
    sys_written_content = "this is a test\n"
    with open(
        sys_written_path, mode="w", encoding="utf-8", newline=""
    ) as sys_written_file:
        sys_written_file.write(sys_written_content)

    with open(bento_model.path_of("sys_written/file"), encoding="utf-8") as f:
        assert f.read() == sys_written_content

    export_tar_path = f"tar://{fs.path.join(str(tmpdir), 'model_b.tar')}"
    bento_model.export(export_tar_path)
    tar_fs = fs.open_fs(export_tar_path)
    from_tar_model = Model.from_fs(tar_fs)

    assert from_tar_model == bento_model
    assert from_tar_model.info == bento_model.info
    assert (
        from_tar_model._fs.readtext("sys_written/file")  # type: ignore
        == sys_written_content
    )
    assert from_tar_model.custom_objects["add"](4) == add_num_1 + 4  # type: ignore
    with pytest.raises(fs.errors.NoSysPath):
        assert from_tar_model.path

    # tmpdir/modelb.bentomodel
    export_bentomodel_path = fs.path.join(str(tmpdir), "modelb.bentomodel")
    bento_model.export(export_bentomodel_path)

    from_fs_model = Model.from_fs(
        fs.tarfs.TarFS(export_bentomodel_path, compression="xz")
    )

    # can cause model.path to fail by using `from_fs`.
    with pytest.raises(fs.errors.NoSysPath):
        assert from_fs_model.path

    model_store = ModelStore(tmpdir)
    from_fs_model.save(model_store)

    save_path = os.path.join(
        tmpdir,
        os.path.join(from_fs_model.tag.name, from_fs_model.tag.version) + os.path.sep,
    )
    assert str(from_fs_model) == f'Model(tag="{bento_model.tag}")'
    assert repr(from_fs_model) == f'Model(tag="{bento_model.tag}", path="{save_path}")'


def test_load_bad_model(tmpdir: "Path"):
    with pytest.raises(BentoMLException):
        Model.from_fs(fs.open_fs(os.path.join(tmpdir, "nonexistent"), create=True))

    bad_path = os.path.join(tmpdir, "badmodel")
    os.makedirs(bad_path)
    with open(
        os.path.join(bad_path, "model.yaml"), "w", encoding="utf-8", newline=""
    ) as model_yaml:
        model_yaml.write("bad yaml")
    with pytest.raises(BentoMLException):
        Model.from_fs(fs.open_fs(bad_path))
