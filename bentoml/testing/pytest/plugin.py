# pylint: disable=unused-argument
from __future__ import annotations

import os
import shutil
import typing as t
import tempfile
import contextlib
from typing import TYPE_CHECKING

import psutil
import pytest
from pytest import MonkeyPatch

from bentoml._internal.utils import LazyLoader
from bentoml._internal.utils import validate_or_create_dir
from bentoml._internal.configuration import CLEAN_BENTOML_VERSION
from bentoml._internal.configuration.containers import BentoMLContainer

if TYPE_CHECKING:
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


@pytest.mark.tryfirst
def pytest_generate_tests(metafunc: Metafunc):
    if "deployment_mode" in metafunc.fixturenames:
        _setup_deployment_mode(metafunc)


@pytest.mark.tryfirst
def pytest_sessionstart(session: Session) -> None:
    """Create a temporary directory for the BentoML home directory, then monkey patch to config."""
    from bentoml._internal.configuration.containers import BentoMLContainer

    mp = MonkeyPatch()
    config = session.config
    config.add_cleanup(mp.undo)
    # setup test environment
    _LOCAL_BUNDLE_BUILD = os.environ.get("BENTOML_BUNDLE_LOCAL_BUILD")
    if _LOCAL_BUNDLE_BUILD:
        # mp this previous value to session to restore to default after test session
        # to avoid affecting local development.
        mp.setattr(
            session,
            "_original_bundle_build",
            _LOCAL_BUNDLE_BUILD,
            raising=False,
        )
    os.environ["BENTOML_BUNDLE_LOCAL_BUILD"] = "True"
    os.environ["SETUPTOOLS_USE_DISTUTILS"] = "stdlib"

    _PYTEST_BENTOML_HOME = tempfile.mkdtemp("bentoml-pytest-e2e")
    bentos = os.path.join(_PYTEST_BENTOML_HOME, "bentos")
    models = os.path.join(_PYTEST_BENTOML_HOME, "models")
    prom_dir = os.path.join(_PYTEST_BENTOML_HOME, "prometheus_multiproc_dir")
    validate_or_create_dir(bentos, models, prom_dir)
    # ensure we setup correct home and prometheus_multiproc_dir folders.
    BentoMLContainer.bentoml_home.set(_PYTEST_BENTOML_HOME)
    BentoMLContainer.prometheus_multiproc_dir.set(prom_dir)
    # setup prometheus multiproc directory for tests.
    _PROMETHEUS_MULTIPROC_DIR = os.environ.get("PROMETHEUS_MULTIPROC_DIR")
    if _PROMETHEUS_MULTIPROC_DIR:
        mp.setattr(
            session,
            "_original_multiproc_env",
            _PROMETHEUS_MULTIPROC_DIR,
            raising=False,
        )
    # use the local bentoml package in development
    os.environ["PROMETHEUS_MULTIPROC_DIR"] = prom_dir

    mp.setattr(config, "_bentoml_home", _PYTEST_BENTOML_HOME, raising=False)


@pytest.mark.tryfirst
@pytest.mark.tryfirst
def pytest_sessionfinish(session: Session, exitstatus: int | ExitCode) -> None:
    config = session.config
    if hasattr(session, "_original_bundle_build"):
        os.environ["BENTOML_BUNDLE_LOCAL_BUILD"] = session._original_bundle_build  # type: ignore (dynamic patch)
    else:
        os.environ.pop("BENTOML_BUNDLE_LOCAL_BUILD", None)
    if hasattr(session, "_original_multiproc_env"):
        os.environ["PROMETHEUS_MULTIPROC_DIR"] = session._original_multiproc_env  # type: ignore (dynamic patch)
    else:
        os.environ.pop("PROMETHEUS_MULTIPROC_DIR", None)
    if config.getoption("cleanup"):
        from bentoml._internal.configuration.containers import BentoMLContainer

        # reset BentoMLContainer.bentoml_home
        BentoMLContainer.bentoml_home.reset()
        # Set dynamically by pytest_configure() above.
        shutil.rmtree(config._bentoml_home)  # type: ignore (dynamic patch)


@pytest.fixture(scope="session")
def bentoml_home(request: FixtureRequest) -> str:
    # Set dynamically by pytest_configure() above.
    return request.config._bentoml_home  # type: ignore (dynamic patch)


@pytest.fixture(scope="session", autouse=True)
def clean_context() -> t.Generator[contextlib.ExitStack, None, None]:
    stack = contextlib.ExitStack()
    yield stack
    stack.close()


@pytest.fixture()
def img_file(tmpdir: str) -> str:
    from PIL.Image import fromarray

    img_file_ = tmpdir.join("test_img.bmp")
    img = fromarray(np.random.randint(255, size=(10, 10, 3)).astype("uint8"))
    img.save(str(img_file_))
    return str(img_file_)


@pytest.fixture()
def bin_file(tmpdir: str) -> str:
    bin_file_ = tmpdir.join("bin_file.bin")
    with open(bin_file_, "wb") as of:
        of.write("â".encode("gb18030"))
    return str(bin_file_)


@pytest.fixture(scope="module", name="metrics_client")
def fixture_metrics_client() -> PrometheusClient:
    return BentoMLContainer.metrics_client.get()
