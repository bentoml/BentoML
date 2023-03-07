import typing as t

import pytest
import cloudpickle

import bentoml
from bentoml.testing.pytest import TEST_MODEL_CONTEXT


@pytest.fixture(scope="session")
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
    def noop_sync(data: str) -> str:  # type: ignore
        return data

    @svc.api(input=Text(), output=Text())
    def invalid(data: str) -> str:  # type: ignore
        raise NotImplementedError

    return svc
