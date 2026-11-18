import importlib.metadata
import os
import random
import string
import time
from sys import version_info as pyver
from typing import TYPE_CHECKING

import pytest

import bentoml
from bentoml._internal.models import ModelContext
from bentoml._internal.models import ModelStore
from bentoml.exceptions import BentoMLException
from bentoml.exceptions import NotFound

if TYPE_CHECKING:
    from pathlib import Path

PYTHON_VERSION: str = f"{pyver.major}.{pyver.minor}.{pyver.micro}"
BENTOML_VERSION: str = importlib.metadata.version("bentoml")


def createfile(filepath: str) -> str:
    content = "".join(random.choices(string.ascii_uppercase + string.digits, k=200))
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, "w", encoding="utf-8") as f:
        f.write(content)
    return content


TEST_MODEL_CONTEXT = ModelContext(
    framework_name="testing", framework_versions={"testing": "v1"}
)


@pytest.mark.parametrize("existing", [False, True])
@pytest.mark.parametrize("trailing_separator", [False, True])
def test_model_folder_export(
    tmp_path: "Path", existing: bool, trailing_separator: bool
):
    store = ModelStore(tmp_path / "models")
    with bentoml.models.create(
        "folder-model:v1", metadata={"score": 0.5}, _model_store=store
    ) as model:
        os.makedirs(model.path_of("nested/empty"))
        with open(model.path_of("nested/weights.bin"), "wb") as f:
            f.write(b"model weights")

    export_dir = tmp_path / "exported"
    if existing:
        export_dir.mkdir()
    export_path = str(export_dir) + (os.sep if trailing_separator else "")

    for iteration in range(2):
        if export_dir.exists():
            export_dir.joinpath("unrelated.txt").write_text("keep me")

        exported = bentoml.models.export_model(
            model.tag, export_path, "folder", _model_store=store
        )
        imported = bentoml.models.import_model(
            exported,
            "folder",
            _model_store=ModelStore(tmp_path / f"imported-{iteration}"),
        )

        assert imported.info == model.info
        assert (
            export_dir.joinpath("nested/weights.bin").read_bytes() == b"model weights"
        )
        assert export_dir.joinpath("nested/empty").is_dir()
        assert not export_dir.joinpath("v1").exists()
        if existing or iteration:
            assert export_dir.joinpath("unrelated.txt").read_text() == "keep me"


def test_models(tmpdir: "Path"):
    os.makedirs(os.path.join(tmpdir, "models"))
    store = ModelStore(os.path.join(tmpdir, "models"))

    with bentoml.models._create(  # type: ignore
        "testmodel",
        module=__name__,
        signatures={},
        context=TEST_MODEL_CONTEXT,
        _model_store=store,
    ) as testmodel:
        testmodel1tag = testmodel.tag

    time.sleep(1)

    with bentoml.models._create(  # type: ignore
        "testmodel",
        module=__name__,
        signatures={},
        context=TEST_MODEL_CONTEXT,
        _model_store=store,
    ) as testmodel:
        testmodel2tag = testmodel.tag
        testmodel_file_content = createfile(testmodel.path_of("file"))
        testmodel_infolder_content = createfile(testmodel.path_of("folder/file"))

    with bentoml.models._create(  # type: ignore
        "anothermodel",
        module=__name__,
        signatures={},
        context=TEST_MODEL_CONTEXT,
        _model_store=store,
    ) as anothermodel:
        anothermodeltag = anothermodel.tag
        anothermodel_file_content = createfile(anothermodel.path_of("file"))
        anothermodel_infolder_content = createfile(anothermodel.path_of("folder/file"))

    with bentoml.models.create(
        "dummymodel",
        _model_store=store,
    ) as dummymodel:
        dummymodel1tag = dummymodel.tag
        dummymodel1_file_content = createfile(dummymodel.path_of("file"))
        dummymodel1_infolder_content = createfile(dummymodel.path_of("folder/file"))

    with bentoml.models.create(
        "dummymodel",
        _model_store=store,
    ) as dummymodel:
        dummymodel2tag = dummymodel.tag
        dummymodel2_file_content = createfile(dummymodel.path_of("file"))
        dummymodel2_infolder_content = createfile(dummymodel.path_of("folder/file"))

    assert (
        bentoml.models.get("testmodel:latest", _model_store=store).tag == testmodel2tag
    )
    assert (
        bentoml.models.get("dummymodel:latest", _model_store=store).tag
        == dummymodel2tag
    )
    assert set([model.tag for model in bentoml.models.list(_model_store=store)]) == {
        testmodel1tag,
        testmodel2tag,
        dummymodel1tag,
        dummymodel2tag,
        anothermodeltag,
    }

    testmodel1 = bentoml.models.get(testmodel1tag, _model_store=store)
    with pytest.raises(FileNotFoundError):
        open(testmodel1.path_of("file"), encoding="utf-8")

    testmodel2 = bentoml.models.get(testmodel2tag, _model_store=store)
    with open(testmodel2.path_of("file"), encoding="utf-8") as f:
        assert f.read() == testmodel_file_content
    with open(testmodel2.path_of("folder/file"), encoding="utf-8") as f:
        assert f.read() == testmodel_infolder_content

    anothermodel = bentoml.models.get(anothermodeltag, _model_store=store)
    with open(anothermodel.path_of("file"), encoding="utf-8") as f:
        assert f.read() == anothermodel_file_content
    with open(anothermodel.path_of("folder/file"), encoding="utf-8") as f:
        assert f.read() == anothermodel_infolder_content

    dummymodel1 = bentoml.models.get(dummymodel1tag, _model_store=store)
    with open(dummymodel1.path_of("file"), encoding="utf-8") as f:
        assert f.read() == dummymodel1_file_content
    with open(dummymodel1.path_of("folder/file"), encoding="utf-8") as f:
        assert f.read() == dummymodel1_infolder_content
    with pytest.raises(BentoMLException):
        dummymodel1.load_model()
    with pytest.raises(BentoMLException):
        dummymodel1.to_runnable()

    dummymodel2 = bentoml.models.get(dummymodel2tag, _model_store=store)
    with open(dummymodel2.path_of("file"), encoding="utf-8") as f:
        assert f.read() == dummymodel2_file_content
    with open(dummymodel2.path_of("folder/file"), encoding="utf-8") as f:
        assert f.read() == dummymodel2_infolder_content
    with pytest.raises(BentoMLException):
        dummymodel1.load_model()
    with pytest.raises(BentoMLException):
        dummymodel1.to_runnable()

    export_path = os.path.join(tmpdir, "testmodel2.bentomodel")
    bentoml.models.export_model(testmodel2tag, export_path, _model_store=store)
    bentoml.models.delete(testmodel2tag, _model_store=store)

    export_path_for_dummymodel = os.path.join(tmpdir, "dummymodel2.bentomodel")
    bentoml.models.export_model(
        dummymodel2tag, export_path_for_dummymodel, _model_store=store
    )
    bentoml.models.delete(dummymodel2tag, _model_store=store)

    with pytest.raises(NotFound):
        bentoml.models.delete(testmodel2tag, _model_store=store)
    with pytest.raises(NotFound):
        bentoml.models.delete(dummymodel2tag, _model_store=store)

    assert set([model.tag for model in bentoml.models.list(_model_store=store)]) == {
        testmodel1tag,
        dummymodel1tag,
        anothermodeltag,
    }

    retrieved_testmodel1 = bentoml.models.get("testmodel", _model_store=store)
    assert retrieved_testmodel1.tag == testmodel1tag
    assert retrieved_testmodel1.info.context.python_version == PYTHON_VERSION
    assert retrieved_testmodel1.info.context.bentoml_version == BENTOML_VERSION
    assert (
        retrieved_testmodel1.info.context.framework_name
        == TEST_MODEL_CONTEXT.framework_name
    )
    assert (
        retrieved_testmodel1.info.context.framework_versions
        == TEST_MODEL_CONTEXT.framework_versions
    )
    assert bentoml.models.get("dummymodel", _model_store=store).tag == dummymodel1tag

    bentoml.models.import_model(export_path, _model_store=store)
    bentoml.models.import_model(export_path_for_dummymodel, _model_store=store)

    assert bentoml.models.get("testmodel", _model_store=store).tag == testmodel2tag
    assert bentoml.models.get("dummymodel", _model_store=store).tag == dummymodel2tag

    export_path_2 = os.path.join(tmpdir, "testmodel1")
    bentoml.models.export_model(testmodel1tag, export_path_2, _model_store=store)
    bentoml.models.delete(testmodel1tag, _model_store=store)
    bentoml.models.import_model(export_path_2 + ".bentomodel", _model_store=store)

    export_path_2_for_dummymodel = os.path.join(tmpdir, "testmodel1")
    bentoml.models.export_model(
        dummymodel1tag, export_path_2_for_dummymodel, _model_store=store
    )
    bentoml.models.delete(dummymodel1tag, _model_store=store)
    bentoml.models.import_model(
        export_path_2_for_dummymodel + ".bentomodel", _model_store=store
    )

    assert bentoml.models.get("testmodel", _model_store=store).tag == testmodel2tag
    assert bentoml.models.get("dummymodel", _model_store=store).tag == dummymodel2tag
