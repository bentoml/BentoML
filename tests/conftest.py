from __future__ import annotations

import os
import typing as t
import pathlib

import yaml
import pytest

import bentoml
from bentoml.io import Text
from bentoml._internal.utils import bentoml_cattr
from bentoml._internal.models import ModelStore
from bentoml._internal.models import ModelContext
from bentoml._internal.bento.build_config import BentoBuildConfig

TEST_MODEL_CONTEXT = ModelContext(
    framework_name="testing", framework_versions={"testing": "v1"}
)


def pytest_generate_tests(metafunc):
    from bentoml._internal.utils import analytics

    analytics.usage_stats.do_not_track.cache_clear()
    analytics.usage_stats.usage_event_debugging.cache_clear()  # type: ignore

    os.environ["__BENTOML_DEBUG_USAGE"] = "False"
    os.environ["BENTOML_DO_NOT_TRACK"] = "True"


@pytest.fixture(scope="function")
def reload_directory(
    tmp_path_factory: pytest.TempPathFactory,
) -> t.Generator[pathlib.Path, None, None]:
    """
    This fixture will create an example bentoml working file directory
    and yield the results directory
    ./
    ├── [fdir, fdir_one, fdir_two]/
    │   ├── README.md
        ├── subdir/
        │   ├── README.md
    │   │   └── app.py
    │   ├── somefile_here.rs
    │   └── app.py
    ├── README.md
    ├── bentofile.yaml
    ├── fname.ipynb
    ├── requirements.txt
    ├── service.py
    └── train.py
    """
    root = tmp_path_factory.mktemp("reload_directory")
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
        description="A simple service",
        include=["*.py"],
        exclude=["*.rs"],
        labels={"foo": "bar", "team": "abc"},
        python=dict(packages=["tensorflow", "numpy"]),
        docker=dict(distro="amazonlinux2"),
    )
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
            ("somefile_here.rs", []),
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


@pytest.fixture(scope="function")
def noop_service() -> bentoml.Service:
    class NoopModel:
        def predict(self, data: t.Any) -> t.Any:
            return data

    bentoml.picklable_model.save_model(
        "noop_model",
        NoopModel(),
        signatures={"predict": {"batchable": True}},
    )

    svc = bentoml.Service(
        name="noop_service",
        runners=[bentoml.picklable_model.get("noop_model").to_runner()],
    )

    @svc.api(input=Text(), output=Text())
    def noop_sync(data: str) -> str:  # type: ignore
        return data

    return svc


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
        "testmodel", signatures={}, context=TEST_MODEL_CONTEXT, _model_store=store
    ):
        pass
    with bentoml.models.create(
        "testmodel", signatures={}, context=TEST_MODEL_CONTEXT, _model_store=store
    ):
        pass
    with bentoml.models.create(
        "anothermodel", signatures={}, context=TEST_MODEL_CONTEXT, _model_store=store
    ):
        pass

    return store
