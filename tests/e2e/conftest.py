# pylint: disable=unused-argument
from __future__ import annotations

import os
import shutil
import typing as t
import tempfile
import contextlib
from typing import TYPE_CHECKING
from importlib import import_module

import psutil
import pytest
from _pytest.monkeypatch import MonkeyPatch

from bentoml.exceptions import InvalidArgument
from bentoml._internal.utils import LazyLoader

if TYPE_CHECKING:

    import numpy as np
    from _pytest.main import Session
    from _pytest.config import Config
    from _pytest.config import ExitCode
    from _pytest.python import Metafunc
    from _pytest.fixtures import FixtureRequest

    class FilledFixtureRequest(FixtureRequest):
        param: str

else:
    np = LazyLoader("np", globals(), "numpy")


def pytest_sessionstart(session: Session) -> None:
    mp = MonkeyPatch()
    session.config.add_cleanup(mp.undo)
    _LOCAL_BUNDLE_BUILD = os.environ.get("BENTOML_BUNDLE_LOCAL_BUILD")
    if _LOCAL_BUNDLE_BUILD:
        # mp this previous value to session to restore to default after test session
        # to avoid affecting local development.
        mp.setattr(
            session,
            "_bentoml_bundle_local_build",
            _LOCAL_BUNDLE_BUILD,
            raising=False,
        )
    # use the local bentoml package in development
    os.environ["BENTOML_BUNDLE_LOCAL_BUILD"] = "True"
    os.environ["SETUPTOOLS_USE_DISTUTILS"] = "stdlib"


def pytest_sessionfinish(session: Session, exitstatus: int | ExitCode) -> None:
    if hasattr(session, "_bentoml_bundle_local_build"):
        os.environ["BENTOML_BUNDLE_LOCAL_BUILD"] = session._bentoml_bundle_local_build  # type: ignore (dynamic patch)
    else:
        os.environ["BENTOML_BUNDLE_LOCAL_BUILD"] = "False"


def pytest_addoption(parser: pytest.Parser):
    parser.addoption("--project-dir", action="store", default=None)
    parser.addoption("--cleanup", action="store_true")


def pytest_generate_tests(metafunc: Metafunc):
    if "deployment_mode" in metafunc.fixturenames:
        if os.getenv("VSCODE_IPC_HOOK_CLI") and not os.getenv("GITHUB_CODESPACE_TOKEN"):
            # When running inside VSCode remote container locally, we don't have access to
            # exposed reserved ports, so we can't run docker-based tests. However on GitHub
            # Codespaces, we can run docker-based tests. (Investigate why this is the case)
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


def pytest_configure(config: Config) -> None:
    """Create a temporary directory for the BentoML home directory, then monkey patch to config."""

    from bentoml._internal.configuration.containers import BentoMLContainer

    _PYTEST_BENTOML_HOME = tempfile.mkdtemp("bentoml-pytest-e2e")
    BentoMLContainer.bentoml_home.set(_PYTEST_BENTOML_HOME)
    bentos = os.path.join(_PYTEST_BENTOML_HOME, "bentos")
    models = os.path.join(_PYTEST_BENTOML_HOME, "models")
    for dir_ in [bentos, models]:
        os.makedirs(dir_, exist_ok=True)

    mp = MonkeyPatch()
    config.add_cleanup(mp.undo)
    mp.setattr(config, "_bentoml_home", _PYTEST_BENTOML_HOME, raising=False)
    project_dir = config.getoption("project_dir")
    assert project_dir, "--project-dir is required"
    try:
        imported = import_module(
            ".configure",
            f"tests.e2e.{t.cast(str, project_dir).rstrip('/').split('/')[-1]}",
        )
        if not hasattr(imported, "create_model"):
            raise InvalidArgument(
                "'create_model()' is required to create a test model."
            ) from None
    except ModuleNotFoundError:
        raise ModuleNotFoundError(
            f"Failed to find 'configure.py' in E2E project '{project_dir}'."
        ) from None
    else:
        imported.create_model()


def pytest_unconfigure(config: Config) -> None:
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
