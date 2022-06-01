from __future__ import annotations

import os
import typing as t
import pathlib

import yaml
import pytest

import bentoml
from bentoml._internal.utils import bentoml_cattr
from bentoml._internal.models import ModelStore
from bentoml._internal.models import ModelContext
from bentoml._internal.bento.build_config import BentoBuildConfig

TEST_MODEL_CONTEXT = ModelContext(
    framework_name="testing", framework_versions={"testing": "v1"}
)


@pytest.fixture(scope="function")
def reload_directory(
    request: pytest.FixtureRequest, tmp_path_factory: pytest.TempPathFactory
) -> t.Generator[pathlib.Path, None, None]:
    """
    This fixture will create an example bentoml working file directory
    and yield the results directory
    ./
    ├── models/  # mock default bentoml home models directory
    ├── [fdir, fdir_one, fdir_two]/
    │   ├── README.md
        ├── subdir/
        │   ├── README.md
    │   │   └── app.py
    │   ├── somerust.rs
    │   └── app.py
    ├── README.md
    ├── .bentoignore
    ├── bentofile.yaml
    ├── fname.ipynb
    ├── requirements.txt
    ├── service.py
    └── train.py
    """
    root = tmp_path_factory.mktemp("reload_directory")
    # create a models directory
    root.joinpath("models").mkdir()

    # enable this fixture to use with unittest.TestCase
    if request.cls is not None:
        request.cls.reload_directory = root

    root_file = [
        "README.md",
        "requirements.txt",
        "service.py",
        "train.py",
        "fname.ipynb",
    ]
    for f in root_file:
        p = root.joinpath(f)
        p.touch()

    build_config = BentoBuildConfig(
        service="service.py:svc",
        description="A mock service",
        exclude=["*.rs"],
    ).with_defaults()
    bentofile = root / "bentofile.yaml"
    bentofile.touch()
    with bentofile.open("w", encoding="utf-8") as f:
        yaml.safe_dump(bentoml_cattr.unstructure(build_config), f)

    custom_library = ["fdir", "fdir_one", "fdir_two"]
    for app in custom_library:
        ap = root.joinpath(app)
        ap.mkdir()
        dir_files: list[tuple[str, list[t.Any]]] = [
            ("README.md", []),
            ("subdir", ["README.md", "app.py"]),
            ("lib.rs", []),
            ("app.py", []),
        ]
        for name, maybe_files in dir_files:
            if maybe_files:
                dpath = ap.joinpath(name)
                dpath.mkdir()
                for f in maybe_files:
                    p = dpath.joinpath(f)
                    p.touch()
            else:
                p = ap.joinpath(name)
                p.touch()

    yield root


@pytest.fixture(scope="function", name="change_test_dir")
def fixture_change_test_dir(
    request: pytest.FixtureRequest,
) -> t.Generator[None, None, None]:
    os.chdir(request.fspath.dirname)  # type: ignore (bad pytest stubs)
    yield
    os.chdir(request.config.invocation_dir)  # type: ignore (bad pytest stubs)


@pytest.fixture(scope="session", name="dummy_model_store")
def fixture_dummy_model_store(tmpdir_factory: "pytest.TempPathFactory") -> ModelStore:
    store = ModelStore(tmpdir_factory.mktemp("models"))
    with bentoml.models.create(
        "testmodel",
        module=__name__,
        signatures={},
        context=TEST_MODEL_CONTEXT,
        _model_store=store,
    ):
        pass
    with bentoml.models.create(
        "testmodel",
        module=__name__,
        signatures={},
        context=TEST_MODEL_CONTEXT,
        _model_store=store,
    ):
        pass
    with bentoml.models.create(
        "anothermodel",
        module=__name__,
        signatures={},
        context=TEST_MODEL_CONTEXT,
        _model_store=store,
    ):
        pass

    return store
