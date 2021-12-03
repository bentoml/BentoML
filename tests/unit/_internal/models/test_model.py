import os
from typing import TYPE_CHECKING
from datetime import datetime
from datetime import timezone

import fs
import pytest
import fs.errors

from bentoml import Tag
from bentoml.exceptions import BentoMLException
from bentoml._internal.models.model import Model
from bentoml._internal.models.model import ModelInfo
from bentoml._internal.configuration import BENTOML_VERSION

if TYPE_CHECKING:
    from pathlib import Path

expected_yaml = """\
name: test
version: version
bentoml_version: {bentoml_version}
creation_time: {creation_time}
api_version: v1
module: testmodule
context:
  ctx: 1
labels:
  label: stringvalue
options:
  option:
    dictkey: dictvalue
metadata:
  metadatakey:
  - list
  - of
  - str
"""


class AdditionClass:
    def __init__(self, x):
        self.x = x

    def __call__(self, y):
        return self.x + y


def test_model_info(tmpdir: "Path"):
    start = datetime.now(timezone.utc)
    modelinfo_a = ModelInfo(Tag("tag"), "module", {}, {}, {}, {})
    end = datetime.now(timezone.utc)

    assert modelinfo_a.bentoml_version == BENTOML_VERSION
    assert modelinfo_a.api_version == "v1"
    assert start <= modelinfo_a.creation_time <= end
    # validate should fail

    tag = Tag("test", "version")
    module = "testmodule"
    labels = {"label": "stringvalue"}
    options = {"option": {"dictkey": "dictvalue"}}
    metadata = {"metadatakey": ["list", "of", "str"]}
    context = {"ctx": 1}
    modelinfo_b = ModelInfo(tag, module, labels, options, metadata, context)

    model_yaml_b_filename = os.path.join(tmpdir, "b_dump.yml")
    with open(model_yaml_b_filename, "w", encoding="utf-8") as model_yaml_b:
        modelinfo_b.dump(model_yaml_b)

    with open(model_yaml_b_filename, encoding="utf-8") as model_yaml_b:
        assert model_yaml_b.read() == expected_yaml.format(
            bentoml_version=BENTOML_VERSION,
            creation_time=modelinfo_b.creation_time.isoformat(" "),
        )

    with open(model_yaml_b_filename, encoding="utf-8") as model_yaml_b:
        modelinfo_b_from_yaml = ModelInfo.from_yaml_file(model_yaml_b)

        assert modelinfo_b_from_yaml == modelinfo_b


def test_model(tmpdir: "Path"):
    start = datetime.now(timezone.utc)
    model_a = Model.create("testmodel")
    end = datetime.now(timezone.utc)

    assert model_a.tag.name == "testmodel"
    assert start <= model_a.creation_time <= end
    assert str(model_a) == f'Model(tag="{model_a.tag}", path="{model_a.path}")'

    add_num_1 = 5
    model_b = Model.create(
        "testmodel1",
        module="test",
        labels={"label": "text"},
        options={"option": "value"},
        framework_context={"ctx": "val"},
        custom_objects={
            "add": AdditionClass(add_num_1),
        },
    )

    # note: models are currently considered to be equal if their tag is equal;
    #       this is a test of that behavior
    eq_to_b = Model.create("tmp")
    eq_to_b._tag = model_b._tag  # type: ignore

    assert eq_to_b == model_b
    assert eq_to_b.__hash__() == model_b.__hash__()

    # note: these tests rely on created models having a system path
    sys_written_path = model_b.path_of("sys_written/file")
    assert sys_written_path == os.path.join(model_b.path, "sys_written", "file")

    os.makedirs(os.path.dirname(sys_written_path))
    sys_written_content = "this is a test\n"
    with open(
        sys_written_path, mode="w", encoding="utf-8", newline=""
    ) as sys_written_file:
        sys_written_file.write(sys_written_content)

    with open(model_b.path_of("sys_written/file"), encoding="utf-8") as f:
        assert f.read() == sys_written_content

    b_tar_path = f"tar://{fs.path.join(str(tmpdir), 'modelb.tar')}"
    model_b.export(b_tar_path)

    tar_fs = fs.open_fs(b_tar_path)

    model_b_from_export = Model.from_fs(tar_fs)

    assert model_b_from_export == model_b
    assert model_b_from_export.info == model_b.info
    assert (
        model_b_from_export._fs.readtext("sys_written/file")  # type: ignore
        == sys_written_content
    )
    assert model_b_from_export.custom_objects["add"](4) == add_num_1 + 4  # type: ignore

    with pytest.raises(fs.errors.NoSysPath):
        assert model_b_from_export.path

    b_export_path = fs.path.join(str(tmpdir), "modelb")
    model_b.export(b_export_path)

    becomes_syspath_model = Model.from_fs(fs.open_fs(b_export_path))

    # check no changes are made when flushing no info changes
    with open(
        becomes_syspath_model.path_of("model.yaml"), encoding="utf-8"
    ) as model_yaml:
        old_yaml = model_yaml.read()
    becomes_syspath_model.flush_info()
    with open(
        becomes_syspath_model.path_of("model.yaml"), encoding="utf-8"
    ) as model_yaml:
        assert model_yaml.read() == old_yaml

    becomes_syspath_model_str = str(becomes_syspath_model)
    sys_export_path = os.path.normpath(b_export_path) + os.path.sep
    assert (
        becomes_syspath_model_str
        == f'Model(tag="{model_b.tag}", path="{sys_export_path}")'
    )
    assert becomes_syspath_model.path is not None

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
