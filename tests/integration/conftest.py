import importlib.util
import types
import typing as t
from typing import TYPE_CHECKING

import pytest

import bentoml.models
from bentoml._internal.models import Model, ModelStore
from tests.utils.helpers import get_model_annotations
from tests.utils.types import InvalidModule, Pipeline

if TYPE_CHECKING:
    from _pytest.config import Config
    from _pytest.config.argparsing import Parser
    from _pytest.nodes import Item
    from _pytest.tmpdir import TempPathFactory

    from bentoml._internal.types import Tag


TEST_MODEL_NAME = __name__.split(".")[-1]


def pytest_addoption(parser: "Parser") -> None:
    parser.addoption(
        "--runslow", action="store_true", default=False, help="run slow tests"
    )
    parser.addoption(
        "--gpus", action="store_true", default=False, help="run gpus related tests"
    )


def pytest_collection_modifyitems(config: "Config", items: t.List["Item"]) -> None:
    if config.getoption("--gpus"):
        return
    skip_gpus = pytest.mark.skip(reason="need --gpus option to run")
    for item in items:
        if "gpus" in item.keywords:
            item.add_marker(skip_gpus)


@pytest.fixture(scope="session", name="modelstore")
def fixture_modelstore(tmp_path_factory: "TempPathFactory") -> "ModelStore":
    # we need to get consistent cache folder, thus tmpdir is not usable here
    # NOTE: after using modelstore, also use `delete_cache_model` to remove model after
    #  load tests.
    path = tmp_path_factory.mktemp("bentoml")
    return ModelStore(path)


@pytest.fixture(scope="session", name="pipeline")
def fixture_pipeline(modelstore: "ModelStore") -> Pipeline:
    def _(
        model: t.Union[t.Callable[..., t.Any], t.Any],
        module: types.ModuleType,
        *args: t.Any,
        name: str = TEST_MODEL_NAME,
        metadata: t.Optional[t.Dict[str, t.Any]] = None,
        return_model: bool = True,
        **kwargs: t.Any,
    ) -> t.Union["Tag", "Model"]:
        if isinstance(model, types.FunctionType):
            model = model()
        assert module is not None and hasattr(module, "__spec__"), "Invalid module"
        spec = getattr(module, "__spec__")
        assert importlib.util.module_from_spec(spec) is not None
        tag = getattr(module, "save")(
            name, model, *args, **kwargs, metadata=metadata, model_store=modelstore
        )
        if return_model:
            return modelstore.get(tag)
        return tag

    return _


@pytest.fixture(scope="session", name="invalid_module")
def fixture_invalid_module(modelstore: "ModelStore") -> InvalidModule:
    def _(save_proc: t.Callable[..., None], *args: t.Any, **kwargs: t.Any) -> str:
        with bentoml.models.create(
            "invalid_module",
            module=__name__,
            labels=None,
            options=None,
            framework_context=None,
            metadata=None,
            _model_store=modelstore,
        ) as models:
            assert isinstance(
                save_proc, types.FunctionType
            ), f"{save_proc.__name__} is not a function"
            if not get_model_annotations(save_proc):
                raise EnvironmentError(
                    f"{save_proc.__name__} requires first argument to of type {type(Model)}"
                )
            save_proc(models, *args, **kwargs)
            return str(models.tag)

    return _
