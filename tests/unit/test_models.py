import os
import time
import random
import string
from sys import version_info as pyver
from typing import TYPE_CHECKING

try:
    import importlib.metadata as importlib_metadata
except ModuleNotFoundError:
    import importlib_metadata

import pytest

import bentoml
from bentoml.exceptions import NotFound
from bentoml._internal.models import ModelStore
from bentoml._internal.models import ModelContext

if TYPE_CHECKING:
    from pathlib import Path

PYTHON_VERSION: str = f"{pyver.major}.{pyver.minor}.{pyver.micro}"
BENTOML_VERSION: str = importlib_metadata.version("bentoml")


def createfile(filepath: str) -> str:
    content = "".join(random.choices(string.ascii_uppercase + string.digits, k=200))
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, "w", encoding="utf-8") as f:
        f.write(content)
    return content


TEST_MODEL_CONTEXT = ModelContext(
    framework_name="testing", framework_versions={"testing": "v1"}
)


def test_models(tmpdir: "Path"):
    os.makedirs(os.path.join(tmpdir, "models"))
    store = ModelStore(os.path.join(tmpdir, "models"))

    with bentoml.models.create(
        "testmodel",
        module=__name__,
        signatures={},
        context=TEST_MODEL_CONTEXT,
        _model_store=store,
    ) as testmodel:
        testmodel1tag = testmodel.tag

    time.sleep(1)

    with bentoml.models.create(
        "testmodel",
        module=__name__,
        signatures={},
        context=TEST_MODEL_CONTEXT,
        _model_store=store,
    ) as testmodel:
        testmodel2tag = testmodel.tag
        testmodel_file_content = createfile(testmodel.path_of("file"))
        testmodel_infolder_content = createfile(testmodel.path_of("folder/file"))

    with bentoml.models.create(
        "anothermodel",
        module=__name__,
        signatures={},
        context=TEST_MODEL_CONTEXT,
        _model_store=store,
    ) as anothermodel:
        anothermodeltag = anothermodel.tag
        anothermodel_file_content = createfile(anothermodel.path_of("file"))
        anothermodel_infolder_content = createfile(anothermodel.path_of("folder/file"))

    assert (
        bentoml.models.get("testmodel:latest", _model_store=store).tag == testmodel2tag
    )
    assert set([model.tag for model in bentoml.models.list(_model_store=store)]) == {
        testmodel1tag,
        testmodel2tag,
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

    export_path = os.path.join(tmpdir, "testmodel2.bentomodel")
    bentoml.models.export_model(testmodel2tag, export_path, _model_store=store)
    bentoml.models.delete(testmodel2tag, _model_store=store)

    with pytest.raises(NotFound):
        bentoml.models.delete(testmodel2tag, _model_store=store)

    assert set([model.tag for model in bentoml.models.list(_model_store=store)]) == {
        testmodel1tag,
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

    bentoml.models.import_model(export_path, _model_store=store)

    assert bentoml.models.get("testmodel", _model_store=store).tag == testmodel2tag

    export_path_2 = os.path.join(tmpdir, "testmodel1")
    bentoml.models.export_model(testmodel1tag, export_path_2, _model_store=store)
    bentoml.models.delete(testmodel1tag, _model_store=store)
    bentoml.models.import_model(export_path_2 + ".bentomodel", _model_store=store)

    assert bentoml.models.get("testmodel", _model_store=store).tag == testmodel2tag
