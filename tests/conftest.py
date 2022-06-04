from __future__ import annotations

import os
import logging
from typing import TYPE_CHECKING

import pytest

import bentoml
from bentoml._internal.models import ModelStore
from bentoml._internal.models import ModelContext

TEST_MODEL_CONTEXT = ModelContext(
    framework_name="testing", framework_versions={"testing": "v1"}
)

if TYPE_CHECKING:
    from _pytest.nodes import Node
    from _pytest.python import Function
    from _pytest.runner import CallInfo
    from _pytest.tmpdir import TempPathFactory
    from _pytest.fixtures import FixtureRequest


# setup @pytest.mark.incremental
def pytest_runtest_makereport(item: Function, call: CallInfo[None]) -> None:
    if "incremental" in item.keywords:
        # call.execinfo is _pytest._code.code.ExceptionInfo
        if call.excinfo is not None and call.excinfo.typename != "Skipped":
            parent = item.parent
            object.__setattr__(parent, "__previousfailed__", item)


def pytest_runtest_setup(item: Function) -> None:
    if "incremental" in item.keywords:
        try:
            previousfailed: Node = object.__getattribute__(
                item.parent, "__previousfailed__"
            )
            pytest.xfail(f"previous test failed ({previousfailed.name})")
        except AttributeError:
            pass


@pytest.fixture(scope="function", name="change_test_dir")
def fixture_change_test_dir(request: FixtureRequest):
    os.chdir(request.fspath.dirname)  # type: ignore (bad pytest stubs)
    yield
    os.chdir(request.config.invocation_dir)  # type: ignore (bad pytest stubs)


@pytest.fixture(scope="session", name="dummy_model_store")
def fixture_dummy_model_store(tmpdir_factory: TempPathFactory) -> ModelStore:
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


@pytest.fixture(scope="function", autouse=True, name="propagate_logs")
def fixture_propagate_logs():
    logger = logging.getLogger("bentoml")
    # bentoml sets propagate to False by default, so we need to set it to True
    # for pytest caplog to recognize logs
    logger.propagate = True
    yield
    # restore propagate to False after tests
    logger.propagate = False
