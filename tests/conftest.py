import os
import typing as t

import pytest

import bentoml
from bentoml._internal.models import ModelStore
from bentoml._internal.models.model import ModelInfo


@pytest.fixture(scope="function", name="change_test_dir")
def fixture_change_test_dir(request: pytest.FixtureRequest):
    os.chdir(request.fspath.dirname)  # type: ignore (bad pytest stubs)
    yield
    os.chdir(request.config.invocation_dir)  # type: ignore (bad pytest stubs)


@pytest.fixture(scope="session", name="dummy_model_store")
def fixture_dummy_model_store(tmpdir_factory: "pytest.TempPathFactory") -> ModelStore:
    store = ModelStore(tmpdir_factory.mktemp("models"))
    with bentoml.models.create("testmodel", _model_store=store):
        pass
    with bentoml.models.create("testmodel", _model_store=store):
        pass
    with bentoml.models.create("anothermodel", _model_store=store):
        pass

    return store


def pytest_assertrepr_compare(
    op: str, left: t.Any, right: t.Any
) -> t.Optional[t.List[str]]:
    if isinstance(left, ModelInfo) and isinstance(right, ModelInfo) and op == "==":
        res = ["Model instances equal:"]

        for attr in [
            "tag",
            "module",
            "labels",
            "options",
            "metadata",
            "context",
            "bentoml_version",
            "api_version",
            "creation_time",
        ]:
            if getattr(left, attr) != getattr(right, attr):
                res.append(
                    "    {attr}: {getattr(left, attr)} != {getattr(right, attr)}"
                )

        return res

    return None
