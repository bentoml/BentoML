import typing as t
from typing import TYPE_CHECKING

from simple_di import inject
from simple_di import Provide

from bentoml import Tag
from bentoml import Model
from bentoml import Runner
from bentoml import SimpleRunner
from bentoml.exceptions import BentoMLException
from bentoml.exceptions import MissingDependencyException

from ..types import PathType
from ..utils import LazyLoader
from ..models import PKL_EXT
from ..models import SAVE_NAMESPACE
from ..utils.pkg import get_pkg_version
from ..configuration.containers import BentoMLContainer

if TYPE_CHECKING:
    import numpy as np
    from .. import ext_typing as ext

    from ..models import ModelStore

import pickle

try:
    import cloudpickle
except ImportError:  # pragma: no cover
    raise MissingDependencyException(
        """cloudpickle is required in order to use the module `bentoml.pickle`, install
         cloudpickle with `pip install cloudpickle`.
        """
    )

MODULE_NAME = "bentoml.pickle"


np = LazyLoader("np", globals(), "numpy")  # noqa: F811
pd = LazyLoader("pd", globals(), "pandas")


def _get_model_info(tag: Tag, model_store: "ModelStore") -> t.Tuple["Model", PathType]:
    model = model_store.get(tag)
    if model.info.module not in (MODULE_NAME, __name__):
        raise BentoMLException(
            f"Model {tag} was saved with module {model.info.module}, failed loading with {MODULE_NAME}."
        )
    model_file = model.path_of(f"{SAVE_NAMESPACE}{PKL_EXT}")

    return model, model_file


@inject
def load(
    tag: t.Union[str, Tag],
    model_store: "ModelStore" = Provide[BentoMLContainer.model_store],
) -> t.Any:
    """
    Load a model from BentoML local modelstore with given name.

    Args:
        tag (:code:`Union[str, Tag]`):
            Tag of a saved model in BentoML local modelstore.
        model_store (:mod:`~bentoml._internal.models.store.ModelStore`, default to :mod:`BentoMLContainer.model_store`):
            BentoML modelstore, provided by DI Container.

    Returns:
        :obj: an instance of :obj: model from BentoML modelstore.

    Examples:

    .. code-block:: python

        import bentoml

        unpickled_model = bentoml.pickle.load('my_model:latest')
    """  # noqa
    _, model_file = _get_model_info(tag, model_store)
    with open(model_file, "rb") as f:
        # The protocol version used is detected automatically, so we do not
        # have to specify it.
        return pickle.load(f)


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
        name (:code:`str`):
            Name for given model instance. This should pass Python identifier check.
        model:
            Instance of model to be saved.
        metadata (:code:`Dict[str, Any]`, `optional`,  default to :code:`None`):
            Custom metadata for given model.
        model_store (:mod:`~bentoml._internal.models.store.ModelStore`, default to :mod:`BentoMLContainer.model_store`):
            BentoML modelstore, provided by DI Container.

    Returns:
        :obj:`~bentoml._internal.types.Tag`: A :obj:`tag` with a format `name:version` where `name` is the user-defined model's name, and a generated `version` by BentoML.

    Examples:

    .. code-block:: python

        import bentoml

        class MyCoolModel:
            def predict(self, some_integer: int):
                return some_integer**2

        model_to_save = MyCoolModel();
        tag_info = bentoml.pickle.save("test_pickle_model", model_to_save)
        runner = bentoml.pickle.load_runner(tag_info)
        runner.run(3)

    """  # noqa
    context = {"framework_name": "pickle"}

    _model = Model.create(
        name,
        module=MODULE_NAME,
        metadata=metadata,
        context=context,
    )

    with open(_model.path_of(f"{SAVE_NAMESPACE}{PKL_EXT}"), "wb") as f:
        cloudpickle.dump(model, f)

    _model.save(model_store)
    return _model.tag


class _PickleRunner(Runner):
    @inject
    def __init__(
        self,
        tag: Tag,
        function_name: str,
        name: str,
        resource_quota: t.Optional[t.Dict[str, t.Any]],
        batch_options: t.Optional[t.Dict[str, t.Any]],
        model_store: "ModelStore" = Provide[BentoMLContainer.model_store],
    ):
        super().__init__(name, resource_quota, batch_options)
        model_info, model_file = _get_model_info(tag, model_store)
        self._model_store = model_store
        self._model_info = model_info
        self._model_file = model_file
        self._function_name = function_name

    @property
    def num_replica(self) -> int:
        return 1

    @property
    def required_models(self) -> t.List[Tag]:
        return [self._model_info.tag]

    # pylint: disable=attribute-defined-outside-init
    def _setup(self) -> None:
        with open(self._model_file, "rb") as f:
            self._model = pickle.load(f)
        self._infer_func = getattr(self._model, self._function_name)

    # pylint: disable=arguments-differ
    def _run_batch(  # type: ignore[reportIncompatibleMethodOverride]
        self,
        inputs: t.Union["ext.NpNDArray", "ext.PdDataFrame"],
    ) -> "ext.NpNDArray":
        return self._infer_func(inputs)


class _PickleSimpleRunner(SimpleRunner):
    @inject
    def __init__(
        self,
        tag: Tag,
        function_name: str,
        name: str,
        resource_quota: t.Optional[t.Dict[str, t.Any]],
        batch_options: t.Optional[t.Dict[str, t.Any]],
        model_store: "ModelStore" = Provide[BentoMLContainer.model_store],
    ):
        super().__init__(name, resource_quota, batch_options)
        model_info, model_file = _get_model_info(tag, model_store)
        self._model_store = model_store
        self._model_info = model_info
        self._model_file = model_file
        self._function_name = function_name

    @property
    def num_replica(self) -> int:
        return 1

    @property
    def required_models(self) -> t.List[Tag]:
        return [self._model_info.tag]

    # pylint: disable=attribute-defined-outside-init
    def _setup(self) -> None:
        with open(self._model_file, "rb") as f:
            self._model = pickle.load(f)
        self._infer_func = getattr(self._model, self._function_name)

    def _run(  # type: ignore[reportIncompatibleMethodOverride]
        self,
        inputs: t.Union["ext.NpNDArray", "ext.PdDataFrame"],
    ) -> "ext.NpNDArray":
        return self._infer_func(inputs)


@inject
def load_runner(
    tag: t.Union[str, Tag],
    function_name: str = "predict",
    *,
    name: t.Optional[str] = None,
    resource_quota: t.Union[None, t.Dict[str, t.Any]] = None,
    batch_options: t.Union[None, t.Dict[str, t.Any]] = None,
    model_store: "ModelStore" = Provide[BentoMLContainer.model_store],
) -> "BaseRunner":
    """
    Runner represents a unit of serving logic that can be scaled horizontally to
    maximize throughput. :func:`bentoml.pickle.load_runner` implements a Runner class that
    wraps the commands that dump and load a pickled object, which optimizes it for the BentoML runtime.

    Args:
        tag (:code:`Union[str, Tag]`):
            Tag of a saved model in BentoML local modelstore..
        function_name (:code:`str`, `optional`, default to :code:`predict`):
            Predict function used by a given pickled model.
        resource_quota (:code:`Dict[str, Any]`, default to :code:`None`):
            Dictionary to configure resources allocation for runner.
        batch_options (:code:`Dict[str, Any]`, default to :code:`None`):
            Dictionary to configure batch options for runner in a service context.
        model_store (:mod:`~bentoml._internal.models.store.ModelStore`, default to :mod:`BentoMLContainer.model_store`):
            BentoML modelstore, provided by DI Container.

    Returns:
        :obj:`~bentoml._internal.runner.Runner`: Runner instances for the target :mod:`bentoml.pickle` model

    Examples:

    .. code-block:: python

        import bentoml

        runner = bentoml.pickle.load_runner("my_model:latest")
        runner.run([[1,2,3,4]])
    """  # noqa
    tag = Tag.from_taglike(tag)
    if name is None:
        name = tag.name

    if batch_options and batch_options.get("enabled"):
        return _PickleRunner(
            tag=tag,
            function_name=function_name,
            name=name,
            resource_quota=resource_quota,
            batch_options=batch_options,
            model_store=model_store,
        )
    else:
        return _PickleSimpleRunner(
            tag=tag,
            function_name=function_name,
            name=name,
            resource_quota=resource_quota,
            batch_options=batch_options,
            model_store=model_store,
        )
