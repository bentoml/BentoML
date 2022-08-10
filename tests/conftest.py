from __future__ import annotations

import os
import typing as t
import logging
import pathlib
from typing import TYPE_CHECKING

import yaml
import pytest

import bentoml
from bentoml._internal.utils import bentoml_cattr
from bentoml._internal.models import ModelStore
from bentoml._internal.models import ModelContext
from bentoml._internal.bento.build_config import BentoBuildConfig

if TYPE_CHECKING:
    from _pytest.python import Metafunc

TEST_MODEL_CONTEXT = ModelContext(
    framework_name="testing", framework_versions={"testing": "v1"}
)


def pytest_generate_tests(metafunc: Metafunc) -> None:
    from bentoml._internal.utils import analytics

    analytics.usage_stats.do_not_track.cache_clear()
    analytics.usage_stats._usage_event_debugging.cache_clear()  # type: ignore (private warning)

    # used for local testing, on CI we already set DO_NOT_TRACK
    os.environ["__BENTOML_DEBUG_USAGE"] = "False"
    os.environ["BENTOML_DO_NOT_TRACK"] = "True"


@pytest.fixture(scope="function")
def noop_service(dummy_model_store: ModelStore) -> bentoml.Service:
    import cloudpickle

    from bentoml.io import Text

    class NoopModel:
        def predict(self, data: t.Any) -> t.Any:
            return data

    with bentoml.models.create(
        "noop_model",
        context=TEST_MODEL_CONTEXT,
        module=__name__,
        signatures={"predict": {"batchable": True}},
        _model_store=dummy_model_store,
    ) as model:
        with open(model.path_of("test.pkl"), "wb") as f:
            cloudpickle.dump(NoopModel(), f)

    ref = bentoml.models.get("noop_model", _model_store=dummy_model_store)

    class NoopRunnable(bentoml.Runnable):
        SUPPORTED_RESOURCES = ("cpu",)
        SUPPORTS_CPU_MULTI_THREADING = True

        def __init__(self):
            self._model: NoopModel = bentoml.picklable_model.load_model(ref)

        @bentoml.Runnable.method(batchable=True)
        def predict(self, data: t.Any) -> t.Any:
            return self._model.predict(data)

    svc = bentoml.Service(
        name="noop_service",
        runners=[bentoml.Runner(NoopRunnable, models=[ref])],
    )

    @svc.api(input=Text(), output=Text())
    def noop_sync(data: str) -> str:  # type: ignore
        return data

    return svc


@pytest.fixture(scope="function", autouse=True, name="propagate_logs")
def fixture_propagate_logs() -> t.Generator[None, None, None]:
    logger = logging.getLogger("bentoml")
    # bentoml sets propagate to False by default, so we need to set it to True
    # for pytest caplog to recognize logs
    logger.propagate = True
    yield
    # restore propagate to False after tests
    logger.propagate = False


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
