import os
import shutil
import typing as t
from typing import TYPE_CHECKING

from simple_di import Provide, inject

from ._internal.configuration.containers import BentoMLContainer
from ._internal.models import SAVE_NAMESPACE, Model
from ._internal.runner import Runner
from ._internal.types import Tag
from .exceptions import MissingDependencyException

if TYPE_CHECKING:  # pragma: no cover
    import numpy as np
    from _internal.models import ModelStore

try:
    from PyRuntime import ExecutionSession
    from PyRuntime import __spec__ as _spec  # pylint: disable=W0622
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
    context: t.Dict[str, t.Any] = {"onnxmlir": _spec.origin}
    _model = Model.create(
        name,
        module=__name__,
        options=None,
        metadata=metadata,
        framework_context=context,
    )
    fpath = _model.path_of(f"{SAVE_NAMESPACE}{ONNXMLIR_EXTENSION}")
    _model.info.options["compiled_path"] = os.path.relpath(fpath, _model.path)
    shutil.copyfile(model, fpath)

    _model.save(model_store)
    return _model.tag


class _ONNXMLirRunner(Runner):
    @inject
    def __init__(
        self,
        tag: t.Union[str, Tag],
        resource_quota: t.Optional[t.Dict[str, t.Any]],
        batch_options: t.Optional[t.Dict[str, t.Any]],
        model_store: "ModelStore" = Provide[BentoMLContainer.model_store],
    ):
        super().__init__(str(tag), resource_quota, batch_options)
        self._model_store = model_store
        self._model_info = model_store.get(tag)

    @property
    def required_models(self) -> t.List[Tag]:
        return [self._model_info.tag]

    @property
    def num_concurrency_per_replica(self) -> int:
        return int(round(self.resource_quota.cpu))

    @property
    def num_replica(self) -> int:
        return 1

    # pylint: disable=attribute-defined-outside-init
    def _setup(self) -> None:
        self._session = load(self.name, self._model_store)

    def _run_batch(
        self, *args: "np.ndarray[t.Any, np.dtype[t.Any]]", **kwargs: t.Any
    ) -> t.List["np.ndarray[t.Any, np.dtype[t.Any]]"]:
        return self._session.run(*args)


@inject
def load_runner(
    tag: t.Union[str, Tag],
    *,
    resource_quota: t.Optional[t.Dict[str, t.Any]] = None,
    batch_options: t.Optional[t.Dict[str, t.Any]] = None,
    model_store: "ModelStore" = Provide[BentoMLContainer.model_store],
) -> "_ONNXMLirRunner":
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
    return _ONNXMLirRunner(
        tag=tag,
        resource_quota=resource_quota,
        batch_options=batch_options,
        model_store=model_store,
    )
