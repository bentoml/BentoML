from __future__ import annotations

import os
import typing as t
import logging

import pytest

import bentoml
from bentoml.io import Text
from bentoml._internal.models import ModelStore
from bentoml._internal.models import ModelContext

TEST_MODEL_CONTEXT = ModelContext(
    framework_name="testing", framework_versions={"testing": "v1"}
)


def pytest_generate_tests(metafunc):
    from bentoml._internal.utils import analytics

    analytics.usage_stats.do_not_track.cache_clear()
    analytics.usage_stats.usage_event_debugging.cache_clear()

    os.environ["__BENTOML_DEBUG_USAGE"] = "False"
    os.environ["BENTOML_DO_NOT_TRACK"] = "True"


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


@pytest.fixture(scope="function", autouse=True, name="propagate_logs")
def fixture_propagate_logs() -> t.Generator[None, None, None]:
    logger = logging.getLogger("bentoml")
    # bentoml sets propagate to False by default, so we need to set it to True
    # for pytest caplog to recognize logs
    logger.propagate = True
    yield
    # restore propagate to False after tests
    logger.propagate = False
