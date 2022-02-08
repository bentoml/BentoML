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

import cloudpickle

MODULE_NAME = "bentoml.pickle"


np = LazyLoader("np", globals(), "numpy")  # noqa: F811
pd = LazyLoader("pd", globals(), "pandas")


def _get_model_info(tag: Tag, model_store: "ModelStore") -> t.Tuple["Model", PathType]:
    model = model_store.get(tag)
    if model.info.module not in (MODULE_NAME, __name__):
        raise BentoMLException(
            f"Model {tag} was saved with module {model.info.module}, cannot load with {MODULE_NAME}."
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
    obj: t.Any,
    batch: bool = False,
    function_name: str = "__call__",
    *,
    metadata: t.Optional[t.Dict[str, t.Any]] = None,
    model_store: "ModelStore" = Provide[BentoMLContainer.model_store],
) -> Tag:
    """
    Save a model instance to BentoML modelstore.

    Args:
        name (:code:`str`):
            Name for given model instance. This should pass Python identifier check.
        obj:
            Instance of an object to be saved.
        batch:
            Determines whether the model supports batching
        function_name:
            Function to call on the pickled object
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
    options = {"batch": batch, "function_name": function_name}

    _model = Model.create(
        name, module=MODULE_NAME, metadata=metadata, context=context, options=options
    )

    with open(_model.path_of(f"{SAVE_NAMESPACE}{PKL_EXT}"), "wb") as f:
        cloudpickle.dump(obj, f)

    _model.save(model_store)
    return _model.tag


class _PickleRunner(Runner):
    @inject
    def __init__(
        self,
        tag: Tag,
        function_name: str,
        name: str,
        model_info: "Model",
        model_file: PathType,
        model_store: "ModelStore" = Provide[BentoMLContainer.model_store],
    ):
        super().__init__(name, {}, {})
        self._model_store = model_store
        self._model_info = model_info
        self._model_file = model_file
        self._function_name = function_name

    @property
    def num_replica(self) -> int:
        return max(int(self.resource_quota.cpu), 1)

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
        inputs: t.Any,
    ) -> t.Any:
        return self._infer_func(inputs)


class _PickleSimpleRunner(SimpleRunner):
    @inject
    def __init__(
        self,
        tag: Tag,
        function_name: str,
        name: str,
        model_info: "Model",
        model_file: PathType,
        model_store: "ModelStore" = Provide[BentoMLContainer.model_store],
    ):
        super().__init__(name, {}, {})
        self._model_store = model_store
        self._model_info = model_info
        self._model_file = model_file
        self._function_name = function_name

    @property
    def num_replica(self) -> int:
        return max(int(self.resource_quota.cpu), 1)

    @property
    def required_models(self) -> t.List[Tag]:
        return [self._model_info.tag]

    # pylint: disable=attribute-defined-outside-init
    def _setup(self) -> None:
        with open(self._model_file, "rb") as f:
            self._model = pickle.load(f)

        if self._function_name == "__call__":
            self._infer_func = self._model
        else:
            self._infer_func = getattr(self._model, self._function_name)

    def _run(  # type: ignore[reportIncompatibleMethodOverride]
        self,
        inputs: t.Any,
    ) -> t.Any:
        return self._infer_func(inputs)


@inject
def load_runner(
    tag: t.Union[str, Tag],
    *,
    name: t.Optional[str] = None,
    model_store: "ModelStore" = Provide[BentoMLContainer.model_store],
) -> "BaseRunner":
    """
    Runner represents a unit of serving logic that can be scaled horizontally to
    maximize throughput. :func:`bentoml.pickle.load_runner` implements a Runner class that
    wraps the commands that dump and load a pickled object, which optimizes it for the BentoML runtime.

    Args:
        tag (:code:`Union[str, Tag]`):
            Tag of a saved model in BentoML local modelstore..
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

    model_info, model_file = _get_model_info(tag, model_store)
    batch_option = model_info.info.options.get("batch")
    function_name = model_info.info.options.get("function_name")

    if batch_option:
        return _PickleRunner(
            tag=tag,
            function_name=function_name,
            name=name,
            model_store=model_store,
            model_info=model_info,
            model_file=model_file,
        )
    else:
        return _PickleSimpleRunner(
            tag=tag,
            function_name=function_name,
            name=name,
            model_store=model_store,
            model_info=model_info,
            model_file=model_file,
        )
