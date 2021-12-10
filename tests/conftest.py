import os

import pytest

import bentoml
from bentoml._internal.models import ModelStore


@pytest.fixture(scope="function", name="change_test_dir")
def fixture_change_test_dir(request: pytest.FixtureRequest):
    os.chdir(request.fspath.dirname)
    yield
    os.chdir(request.config.invocation_dir)


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
