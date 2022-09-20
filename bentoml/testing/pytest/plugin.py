# pylint: disable=unused-argument
from __future__ import annotations

import os
import shutil
import typing as t
import logging
import tempfile
import contextlib
from typing import TYPE_CHECKING

import yaml
import psutil
import pytest
import cloudpickle
from pytest import MonkeyPatch

import bentoml
from bentoml._internal.utils import LazyLoader
from bentoml._internal.utils import validate_or_create_dir
from bentoml._internal.models import ModelContext
from bentoml._internal.configuration import CLEAN_BENTOML_VERSION
from bentoml._internal.configuration.containers import BentoMLContainer

if TYPE_CHECKING:
    from pathlib import Path

    import numpy as np
    from _pytest.main import Session
    from _pytest.main import PytestPluginManager  # type: ignore (not exported warning)
    from _pytest.config import Config
    from _pytest.config import ExitCode
    from _pytest.python import Metafunc
    from _pytest.fixtures import FixtureRequest
    from _pytest.config.argparsing import Parser

    class FilledFixtureRequest(FixtureRequest):
        param: str

    from bentoml._internal.server.metrics.prometheus import PrometheusClient

else:
    np = LazyLoader("np", globals(), "numpy")


TEST_MODEL_CONTEXT = ModelContext(
    framework_name="testing",
    framework_versions={"testing": "v1"},
)


@pytest.mark.tryfirst
def pytest_report_header(config: Config) -> list[str]:
    return [f"bentoml: version={CLEAN_BENTOML_VERSION}"]


@pytest.mark.tryfirst
def pytest_addoption(parser: Parser, pluginmanager: PytestPluginManager) -> None:
    group = parser.getgroup("bentoml")
    group.addoption(
        "--cleanup",
        action="store_true",
        help="If passed, We will cleanup temporary directory after session is finished.",
    )


def _setup_deployment_mode(metafunc: Metafunc):
    """
    Setup deployment mode for test session.
    We will dynamically add this fixture to tests functions that has ``deployment_mode`` fixtures.

    Current matrix:
    - deployment_mode: ["docker", "distributed", "standalone"]
    """
    if os.getenv("VSCODE_IPC_HOOK_CLI") and not os.getenv("GITHUB_CODESPACE_TOKEN"):
        # When running inside VSCode remote container locally, we don't have access to
        # exposed reserved ports, so we can't run docker-based tests. However on GitHub
        # Codespaces, we can run docker-based tests.
        # Note that inside the remote container, it is already running as a Linux container.
        deployment_mode = ["distributed", "standalone"]
    else:
        if os.environ.get("GITHUB_ACTIONS") and (psutil.WINDOWS or psutil.MACOS):
            # Due to GitHub Actions' limitation, we can't run docker-based tests
            # on Windows and macOS. However, we can still running those tests on
            # local development.
            if psutil.MACOS:
                deployment_mode = ["distributed", "standalone"]
            else:
                deployment_mode = ["standalone"]
        else:
            if psutil.WINDOWS:
                deployment_mode = ["standalone", "docker"]
            else:
                deployment_mode = ["distributed", "standalone", "docker"]
    metafunc.parametrize("deployment_mode", deployment_mode, scope="session")


def _setup_model_store(metafunc: Metafunc):
    """Setup dummy models for test session."""
    with bentoml.models.create(
        "testmodel",
        module=__name__,
        signatures={},
        context=TEST_MODEL_CONTEXT,
    ):
        pass
    with bentoml.models.create(
        "testmodel",
        module=__name__,
        signatures={},
        context=TEST_MODEL_CONTEXT,
    ):
        pass
    with bentoml.models.create(
        "anothermodel",
        module=__name__,
        signatures={},
        context=TEST_MODEL_CONTEXT,
    ):
        pass

    metafunc.parametrize(
        "model_store", [BentoMLContainer.model_store.get()], scope="session"
    )


@pytest.mark.tryfirst
def pytest_generate_tests(metafunc: Metafunc):
    if "deployment_mode" in metafunc.fixturenames:
        _setup_deployment_mode(metafunc)
    if "model_store" in metafunc.fixturenames:
        _setup_model_store(metafunc)


def _setup_session_environment(
    mp: MonkeyPatch, o: Session | Config, *pairs: tuple[str, str]
):
    """Setup environment variable for test session."""
    for p in pairs:
        key, value = p
        _ENV_VAR = os.environ.get(key)
        if _ENV_VAR:
            mp.setattr(o, f"_original_{key}", _ENV_VAR, raising=False)
        os.environ[key] = value


@pytest.mark.tryfirst
def pytest_sessionstart(session: Session) -> None:
    """Create a temporary directory for the BentoML home directory, then monkey patch to config."""
    from bentoml._internal.utils import analytics

    # We need to clear analytics cache before running tests.
    analytics.usage_stats.do_not_track.cache_clear()
    analytics.usage_stats._usage_event_debugging.cache_clear()  # type: ignore (private warning)

    mp = MonkeyPatch()
    config = session.config
    config.add_cleanup(mp.undo)

    # Ensure we setup correct home and prometheus_multiproc_dir folders.
    # For any given test session.
    _PYTEST_BENTOML_HOME = tempfile.mkdtemp("bentoml-pytest")
    _PYTEST_MULTIPROC_DIR = os.path.join(
        _PYTEST_BENTOML_HOME, "prometheus_multiproc_dir"
    )
    validate_or_create_dir(
        *[
            os.path.join(_PYTEST_BENTOML_HOME, d)
            for d in ["bentos", "models", "prometheus_multiproc_dir"]
        ]
    )
    BentoMLContainer.bentoml_home.set(_PYTEST_BENTOML_HOME)
    BentoMLContainer.prometheus_multiproc_dir.set(_PYTEST_MULTIPROC_DIR)

    # Ensure that we will always build bento using bentoml from source
    # Setup prometheus multiproc directory for tests.
    _setup_session_environment(
        mp,
        session,
        ("PROMETHEUS_MULTIPROC_DIR", _PYTEST_MULTIPROC_DIR),
        ("BENTOML_BUNDLE_LOCAL_BUILD", "True"),
        ("SETUPTOOLS_USE_DISTUTILS", "stdlib"),
        ("__BENTOML_DEBUG_USAGE", "False"),
        ("BENTOML_DO_NOT_TRACK", "True"),
    )

    _setup_session_environment(mp, config, ("BENTOML_HOME", _PYTEST_BENTOML_HOME))


def _teardown_session_environment(session: Session, *variables: str):
    """Restore environment variable to original value."""
    for variable in variables:
        if hasattr(session, f"_original_{variable}"):
            os.environ[variable] = getattr(session, f"_original_{variable}")
        else:
            os.environ.pop(variable, None)


@pytest.mark.tryfirst
def pytest_sessionfinish(session: Session, exitstatus: int | ExitCode) -> None:
    config = session.config

    # reset home and prometheus_multiproc_dir to default
    BentoMLContainer.bentoml_home.reset()
    BentoMLContainer.prometheus_multiproc_dir.reset()

    _teardown_session_environment(
        session,
        "BENTOML_BUNDLE_LOCAL_BUILD",
        "PROMETHEUS_MULTIPROC_DIR",
        "SETUPTOOLS_USE_DISTUTILS",
        "__BENTOML_DEBUG_USAGE",
        "BENTOML_DO_NOT_TRACK",
    )
    if config.getoption("cleanup"):
        # Set dynamically by pytest_configure() above.
        shutil.rmtree(config._bentoml_home)  # type: ignore (dynamic patch)


@pytest.fixture(scope="session")
def bentoml_home(request: FixtureRequest) -> str:
    """
    Return the BentoML home directory for the test session.
    This directory is created via ``pytest_sessionstart``.
    """
    # Set dynamically by pytest_configure() above.
    return request.config._bentoml_home  # type: ignore (dynamic patch)


@pytest.fixture(scope="session", autouse=True)
def clean_context() -> t.Generator[contextlib.ExitStack, None, None]:
    """
    Create a ExitStack to cleanup contextmanager.
    This fixture is available to all tests.
    """
    stack = contextlib.ExitStack()
    yield stack
    stack.close()


@pytest.fixture()
def img_file(tmpdir: str) -> str:
    """Create a random image/bmp file."""
    from PIL.Image import fromarray

    img_file_ = tmpdir.join("test_img.bmp")
    img = fromarray(np.random.randint(255, size=(10, 10, 3)).astype("uint8"))
    img.save(str(img_file_))
    return str(img_file_)


@pytest.fixture()
def bin_file(tmpdir: str) -> str:
    """Create a random binary file."""
    bin_file_ = tmpdir.join("bin_file.bin")
    with open(bin_file_, "wb") as of:
        of.write("â".encode("gb18030"))
    return str(bin_file_)


@pytest.fixture(scope="module", name="metrics_client")
def fixture_metrics_client() -> PrometheusClient:
    """This fixtures return a PrometheusClient instance that can be used for testing."""
    return BentoMLContainer.metrics_client.get()


@pytest.fixture(scope="function")
def reload_directory(
    request: FilledFixtureRequest, tmp_path_factory: pytest.TempPathFactory
) -> t.Generator[Path, None, None]:
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
    from bentoml._internal.utils import bentoml_cattr
    from bentoml._internal.bento.build_config import BentoBuildConfig

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


@pytest.fixture(scope="module")
def simple_service() -> bentoml.Service:
    """
    This fixture create a simple service implementation that implements a noop runnable with two APIs:

    - noop_sync: sync API that returns the input.
    - invalid: an invalid API that can be used to test error handling.
    """
    from bentoml.io import Text

    class NoopModel:
        def predict(self, data: t.Any) -> t.Any:
            return data

    with bentoml.models.create(
        "python_function",
        context=TEST_MODEL_CONTEXT,
        module=__name__,
        signatures={"predict": {"batchable": True}},
    ) as model:
        with open(model.path_of("test.pkl"), "wb") as f:
            cloudpickle.dump(NoopModel(), f)

    model_ref = bentoml.models.get("python_function")

    class NoopRunnable(bentoml.Runnable):
        SUPPORTED_RESOURCES = ("cpu",)
        SUPPORTS_CPU_MULTI_THREADING = True

        def __init__(self):
            self._model: NoopModel = bentoml.picklable_model.load_model(model_ref)

        @bentoml.Runnable.method(batchable=True)
        def predict(self, data: t.Any) -> t.Any:
            return self._model.predict(data)

    svc = bentoml.Service(
        name="simple_service",
        runners=[bentoml.Runner(NoopRunnable, models=[model_ref])],
    )

    @svc.api(input=Text(), output=Text())
    def noop_sync(data: str) -> str:
        return data

    @svc.api(input=Text(), output=Text())
    def invalid(data: str) -> str:
        raise RuntimeError("invalid implementation.")

    return svc


@pytest.fixture(scope="function", name="propagate_logs")
def fixture_propagate_logs() -> t.Generator[None, None, None]:
    """BentoML sets propagate to False by default, hence this fixture enable log propagation."""
    logger = logging.getLogger("bentoml")
    logger.propagate = True
    yield
    # restore propagate to False after tests
    logger.propagate = False


@pytest.fixture(scope="function", name="change_test_dir")
def fixture_change_dir(request: pytest.FixtureRequest) -> t.Generator[None, None, None]:
    """A fixture to change given test directory to the directory of the current running test."""
    os.chdir(request.fspath.dirname)  # type: ignore (bad pytest stubs)
    yield
    os.chdir(request.config.invocation_dir)  # type: ignore (bad pytest stubs)
