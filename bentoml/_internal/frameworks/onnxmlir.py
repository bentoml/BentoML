import os
import shutil
import typing as t
from typing import TYPE_CHECKING

import attr
import numpy as np
from simple_di import inject
from simple_di import Provide

from bentoml import Tag
from bentoml import Runner
from bentoml.exceptions import MissingDependencyException

from ..models import Model
from ..models import SAVE_NAMESPACE
from ..configuration.containers import BentoMLContainer

if TYPE_CHECKING:
    from ..models import ModelStore

try:
    from PyRuntime import __spec__ as _spec  # pylint: disable=W0622
    from PyRuntime import ExecutionSession
except ImportError:  # pragma: no cover
    raise MissingDependencyException(
        """\
PyRuntime is not found in PYTHONPATH. Refers to
 https://github.com/onnx/onnx-mlir#installation-on-unix for
 more information.
    """
    )

ONNXMLIR_EXTENSION: str = ".so"


@inject
def load(
    tag: t.Union[str, Tag],
    model_store: "ModelStore" = Provide[BentoMLContainer.model_store],
) -> "ExecutionSession":
    """
    Load a model from BentoML local modelstore with given name.

    onnx-mlir is a compiler technology that can take an onnx model and lower it
    (using llvm) to an inference library that is optimized and has little external
    dependencies.

    The PyRuntime interface is created during the build of onnx-mlir using pybind.
    See the onnx-mlir supporting documentation for detail.

    Args:
        tag (`str`):
            Tag of a saved model in BentoML local modelstore.
        model_store (`~bentoml._internal.models.store.ModelStore`, default to `BentoMLContainer.model_store`):
            BentoML modelstore, provided by DI Container.

    Returns:
        an instance of `xgboost.core.Booster` from BentoML modelstore.

    Examples::
    """  # noqa
    model = model_store.get(tag)
    compiled_path = model.path_of(model.info.options["compiled_path"])
    return ExecutionSession(compiled_path, "run_main_graph")


@inject
def save(
    name: str,
    model: t.Any,
    *,
    metadata: t.Optional[t.Dict[str, t.Any]] = None,
    model_store: "ModelStore" = Provide[BentoMLContainer.model_store],
) -> Tag:
    """
    Save a model instance to BentoML modelstore.

    Args:
        name (`str`):
            Name for given model instance. This should pass Python identifier check.
        model (`str`):
            Path to compiled model by MLir
        metadata (`t.Optional[t.Dict[str, t.Any]]`, default to `None`):
            Custom metadata for given model.
        model_store (`~bentoml._internal.models.store.ModelStore`, default to `BentoMLContainer.model_store`):
            BentoML modelstore, provided by DI Container.

    Returns:
        tag (`str` with a format `name:version`) where `name` is the defined name user
        set for their models, and version will be generated by BentoML.

    Examples::
    """  # noqa
    context: t.Dict[str, t.Any] = {
        "framework_name": "onnxmlir",
        "onnxmlir_version": _spec.origin,
    }
    _model = Model.create(
        name,
        module=__name__,
        options=None,
        metadata=metadata,
        context=context,
    )
    fpath = _model.path_of(f"{SAVE_NAMESPACE}{ONNXMLIR_EXTENSION}")
    _model.info.options["compiled_path"] = os.path.relpath(fpath, _model.path)
    shutil.copyfile(model, fpath)

    _model.save(model_store)
    return _model.tag


class ONNXMLirRunner(Runner):
    @inject
    def __init__(
        self,
        tag: t.Union[str, Tag],
        resource_quota: t.Optional[t.Dict[str, t.Any]],
        batch_options: t.Optional[t.Dict[str, t.Any]],
        model_store: "ModelStore" = Provide[BentoMLContainer.model_store],
    ):
        self._model_store = model_store
        self._model_tag = Tag.from_taglike(tag)
        name = f"{self.__class__.__name__}_{self._model_tag.name}"
        super().__init__(name, resource_quota, batch_options)

    @property
    def required_models(self) -> t.List[Tag]:
        return [self._model_tag]

    @property
    def num_concurrency_per_replica(self) -> int:
        return int(round(self.resource_quota.cpu))

    @property
    def num_replica(self) -> int:
        return 1

    # pylint: disable=arguments-differ,attribute-defined-outside-init
    def _setup(self) -> None:  # type: ignore[override]
        self._session = load(self._model_tag, self._model_store)

    # pylint: disable=arguments-differ
    def _run_batch(self, input_data: np.ndarray) -> np.ndarray:  # type: ignore[override] # noqa: LN001
        return self._session.run(input_data)


@inject
def load_runner(
    tag: t.Union[str, Tag],
    *,
    resource_quota: t.Optional[t.Dict[str, t.Any]] = None,
    batch_options: t.Optional[t.Dict[str, t.Any]] = None,
    model_store: "ModelStore" = Provide[BentoMLContainer.model_store],
) -> "ONNXMLirRunner":
    """
    Runner represents a unit of serving logic that can be scaled horizontally to
    maximize throughput. `bentoml.xgboost.load_runner` implements a Runner class that
    wrap around a Xgboost booster model, which optimize it for the BentoML runtime.

    Args:
        tag (`str`):
            Model tag to retrieve model from modelstore
        resource_quota (`t.Dict[str, t.Any]`, default to `None`):
            Dictionary to configure resources allocation for runner.
        batch_options (`t.Dict[str, t.Any]`, default to `None`):
            Dictionary to configure batch options for runner in a service context.
        model_store (`~bentoml._internal.models.store.ModelStore`, default to `BentoMLContainer.model_store`):
            BentoML modelstore, provided by DI Container.

    Returns:
        Runner instances for `bentoml.xgboost` model

    Examples::
    """  # noqa
    return ONNXMLirRunner(
        tag=tag,
        resource_quota=resource_quota,
        batch_options=batch_options,
        model_store=model_store,
    )
