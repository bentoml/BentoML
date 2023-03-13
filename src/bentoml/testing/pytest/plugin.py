# pylint: disable=unused-argument
from __future__ import annotations

import os
import typing as t
import tempfile
import contextlib
from typing import TYPE_CHECKING

import psutil
import pytest
from pytest import MonkeyPatch

import bentoml
from bentoml._internal.utils import LazyLoader
from bentoml._internal.utils import validate_or_create_dir
from bentoml._internal.models import ModelContext
from bentoml._internal.configuration import CLEAN_BENTOML_VERSION
from bentoml._internal.configuration.containers import BentoMLContainer

if TYPE_CHECKING:
    import numpy as np
    from _pytest.main import Session
    from _pytest.nodes import Item
    from _pytest.config import Config
    from _pytest.config import ExitCode
    from _pytest.python import Metafunc
    from _pytest.fixtures import FixtureRequest
    from _pytest.config.argparsing import Parser

    from bentoml._internal.server.metrics.prometheus import PrometheusClient

else:
    np = LazyLoader("np", globals(), "numpy")


TEST_MODEL_CONTEXT = ModelContext(
    framework_name="testing",
    framework_versions={"testing": "v1"},
)

_RUN_GPU_TESTS_MARKER = "--run-gpu-tests"
_RUN_GRPC_TESTS_MARKER = "--run-grpc-tests"


@pytest.mark.tryfirst
def pytest_report_header(config: Config) -> list[str]:
    return [f"bentoml: version={CLEAN_BENTOML_VERSION}"]


@pytest.hookimpl
def pytest_addoption(parser: Parser) -> None:
    group = parser.getgroup("bentoml", "BentoML pytest plugins.")
    group.addoption(
        _RUN_GPU_TESTS_MARKER,
        action="store_true",
        default=False,
        help="run gpus related tests.",
    )
    group.addoption(
        _RUN_GRPC_TESTS_MARKER,
        action="store_true",
        default=False,
        help="run grpc related tests.",
    )


def pytest_configure(config: Config) -> None:
    # We will inject marker documentation here.
    config.addinivalue_line(
        "markers",
        "requires_gpus: requires GPU to run given test.",
    )
    config.addinivalue_line(
        "markers",
        "requires_grpc: requires gRPC support to run given test.",
    )


@pytest.hookimpl(tryfirst=True)
def pytest_runtest_setup(item: Item) -> None:
    config = item.config
    if "requires_gpus" in item.keywords and not config.getoption(_RUN_GPU_TESTS_MARKER):
        item.add_marker(
            pytest.mark.skip(
                reason=f"need {_RUN_GPU_TESTS_MARKER} option to run gpus related tests."
            )
        )
    # We don't run gRPC tests on Windows
    if "requires_grpc" in item.keywords and not config.getoption(
        _RUN_GRPC_TESTS_MARKER
    ):
        item.add_marker(
            pytest.mark.skip(
                reason=f"need {_RUN_GRPC_TESTS_MARKER} option to run grpc related tests."
            )
        )


def _setup_deployment_mode(metafunc: Metafunc):
    """
    Setup deployment mode for test session.
    We will dynamically add this fixture to tests functions that has ``deployment_mode`` fixtures.

    Current matrix:
    - deployment_mode: ["container", "distributed", "standalone"]
    """
    if os.getenv("VSCODE_IPC_HOOK_CLI") and not os.getenv("GITHUB_CODESPACE_TOKEN"):
        # When running inside VSCode remote container locally, we don't have access to
        # exposed reserved ports, so we can't run container-based tests. However on GitHub
        # Codespaces, we can run container-based tests.
        # Note that inside the remote container, it is already running as a Linux container.
        deployment_mode = ["distributed", "standalone"]
    else:
        if os.environ.get("GITHUB_ACTIONS") and (psutil.WINDOWS or psutil.MACOS):
            # Due to GitHub Actions' limitation, we can't run container-based tests
            # on Windows and macOS. However, we can still running those tests on
            # local development.
            if psutil.MACOS:
                deployment_mode = ["distributed", "standalone"]
            else:
                deployment_mode = ["standalone"]
        else:
            if psutil.WINDOWS:
                deployment_mode = ["standalone", "container"]
            else:
                deployment_mode = ["distributed", "standalone", "container"]
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
        _ENV_VAR = os.environ.get(key, None)
        if _ENV_VAR is not None:
            mp.setattr(o, f"_original_{key}", _ENV_VAR, raising=False)
        os.environ[key] = value


def _setup_test_directory() -> tuple[str, str]:
    # Ensure we setup correct home and prometheus_multiproc_dir folders.
    # For any given test session.
    bentoml_home = tempfile.mkdtemp("bentoml-pytest")
    bentos = os.path.join(bentoml_home, "bentos")
    models = os.path.join(bentoml_home, "models")
    tmp_bentos = os.path.join(bentoml_home, "tmp_bentos")
    multiproc_dir = os.path.join(bentoml_home, "prometheus_multiproc_dir")
    validate_or_create_dir(bentos, models, tmp_bentos, multiproc_dir)

    # We need to set the below value inside container due to
    # the fact that each value is a singleton, and will be cached.
    BentoMLContainer.bentoml_home.set(bentoml_home)
    BentoMLContainer.bento_store_dir.set(bentos)
    BentoMLContainer.model_store_dir.set(models)
    BentoMLContainer.tmp_bento_store_dir.set(tmp_bentos)
    BentoMLContainer.prometheus_multiproc_dir.set(multiproc_dir)
    return bentoml_home, multiproc_dir


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

    _PYTEST_BENTOML_HOME, _PYTEST_MULTIPROC_DIR = _setup_test_directory()

    # The evironment variable patch ensures that we will
    # always build bento using bentoml from source, use the correct
    # test bentoml home directory, and setup prometheus multiproc directory.
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


def _teardown_session_environment(o: Session | Config, *variables: str):
    """Restore environment variable to original value."""
    for variable in variables:
        if hasattr(o, f"_original_{variable}"):
            os.environ[variable] = getattr(o, f"_original_{variable}")
        else:
            os.environ.pop(variable, None)


@pytest.mark.tryfirst
def pytest_sessionfinish(session: Session, exitstatus: int | ExitCode) -> None:
    config = session.config

    _teardown_session_environment(
        session,
        "BENTOML_BUNDLE_LOCAL_BUILD",
        "PROMETHEUS_MULTIPROC_DIR",
        "SETUPTOOLS_USE_DISTUTILS",
        "__BENTOML_DEBUG_USAGE",
        "BENTOML_DO_NOT_TRACK",
    )
    _teardown_session_environment(config, "BENTOML_HOME")

    # reset home and prometheus_multiproc_dir to default
    BentoMLContainer.prometheus_multiproc_dir.reset()


@pytest.fixture(scope="session")
def bentoml_home() -> str:
    """
    Return the BentoML home directory for the test session.
    This directory is created via ``pytest_sessionstart``.
    """
    return BentoMLContainer.bentoml_home.get()


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


@pytest.fixture(scope="module", name="prom_client")
def fixture_metrics_client() -> PrometheusClient:
    """This fixtures return a PrometheusClient instance that can be used for testing."""
    return BentoMLContainer.metrics_client.get()


@pytest.fixture(scope="function", name="change_test_dir")
def fixture_change_dir(request: FixtureRequest) -> t.Generator[None, None, None]:
    """A fixture to change given test directory to the directory of the current running test."""
    os.chdir(request.fspath.dirname)  # type: ignore (bad pytest stubs)
    yield
    os.chdir(request.config.invocation_dir)  # type: ignore (bad pytest stubs)
