import os

import pytest

import bentoml
from bentoml._internal.models import ModelStore
from bentoml._internal.models import ModelContext

TEST_MODEL_CONTEXT = ModelContext(
    framework_name="testing", framework_versions={"testing": "v1"}
)


@pytest.fixture(scope="function", name="change_test_dir")
def fixture_change_test_dir(request: pytest.FixtureRequest):
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
