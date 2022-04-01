import pickle
import typing as t
from typing import TYPE_CHECKING

import cloudpickle  # type: ignore
from simple_di import inject
from simple_di import Provide

import bentoml
from bentoml import Tag
from bentoml.exceptions import BentoMLException

from ..models import PKL_EXT
from ..models import SAVE_NAMESPACE
from .common.model_runner import BaseModelRunner
from .common.model_runner import BaseModelSimpleRunner
from ..configuration.containers import BentoMLContainer

if TYPE_CHECKING:
    from ..models import ModelStore

MODULE_NAME = "bentoml.picklable_model"


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

        unpickled_model = bentoml.picklable_model.load('my_model:latest')
    """
    model = model_store.get(tag)

    if model.info.module not in (MODULE_NAME, __name__):
        raise BentoMLException(
            f"Model {tag} was saved with module {model.info.module}, cannot load with {MODULE_NAME}."
        )

    model_file = model.path_of(f"{SAVE_NAMESPACE}{PKL_EXT}")
    with open(model_file, "rb") as f:
        # The protocol version used is detected automatically, so we do not
        # have to specify it.
        return pickle.load(f)


def save(
    name: str,
    obj: t.Any,
    *,
    labels: t.Optional[t.Dict[str, str]] = None,
    custom_objects: t.Optional[t.Dict[str, t.Any]] = None,
    metadata: t.Optional[t.Dict[str, t.Any]] = None,
) -> Tag:
    """
    Save a model instance to BentoML modelstore.

    Args:
        name (:code:`str`):
            Name for given model instance. This should pass Python identifier check.
        obj:
            Instance of an object to be saved.
        labels (:code:`Dict[str, str]`, `optional`, default to :code:`None`):
            user-defined labels for managing models, e.g. team=nlp, stage=dev
        custom_objects (:code:`Dict[str, Any]]`, `optional`, default to :code:`None`):
            user-defined additional python objects to be saved alongside the model,
            e.g. a tokenizer instance, preprocessor function, model configuration json
        metadata (:code:`Dict[str, Any]`, `optional`,  default to :code:`None`):
            Custom metadata for given model.
        model_store (:mod:`~bentoml._internal.models.store.ModelStore`, default to :mod:`BentoMLContainer.model_store`):
            BentoML modelstore, provided by DI Container.

    Returns:
        :obj:`~bentoml.Tag`: A :obj:`tag` with a format `name:version` where `name` is the user-defined model's name, and a generated `version` by BentoML.

    Examples:

    .. code-block:: python

        import bentoml

        class MyCoolModel:
            def predict(self, some_integer: int):
                return some_integer**2

        model_to_save = MyCoolModel();
        tag_info = bentoml.picklable_model.save("test_pickle_model", model_to_save)
        runner = bentoml.picklable_model.load_runner(tag_info)
        runner.run(3)

    """
    context = {"framework_name": "picklable_model"}

    with bentoml.models.create(
        name,
        module=MODULE_NAME,
        labels=labels,
        custom_objects=custom_objects,
        metadata=metadata,
        context=context,
    ) as _model:

        with open(_model.path_of(f"{SAVE_NAMESPACE}{PKL_EXT}"), "wb") as f:
            cloudpickle.dump(obj, f)

        return _model.tag


class _PicklableModelRunner(BaseModelRunner):
    def __init__(self, tag: t.Union[Tag, str], method_name: str, name: t.Optional[str]):
        super().__init__(tag=tag, name=name)

        self._method_name = method_name

        self._model: t.Any = None
        self._infer_func: t.Any = None

    @property
    def num_replica(self) -> int:
        return max(round(self.resource_quota.cpu), 1)

    def _setup(self) -> None:
        self._model = load(self._tag, model_store=self.model_store)
        if self._method_name == "__call__":
            self._infer_func = self._model
        else:
            self._infer_func = getattr(self._model, self._method_name)

    def _run_batch(self, *args: t.Any, **kwargs: t.Any) -> t.Any:
        return self._infer_func(*args, **kwargs)


class _PicklableModelSimpleRunner(BaseModelSimpleRunner):
    def __init__(self, tag: t.Union[Tag, str], method_name: str, name: t.Optional[str]):
        super().__init__(tag=tag, name=name)
        self._method_name = method_name

        self._model: t.Any = None
        self._infer_func: t.Any = None

    @property
    def num_replica(self) -> int:
        return max(round(self.resource_quota.cpu), 1)

    def _setup(self) -> None:
        self._model = load(self._tag, model_store=self.model_store)

        if self._method_name == "__call__":
            self._infer_func = self._model
        else:
            self._infer_func = getattr(self._model, self._method_name)

    def _run(self, *args: t.Any, **kwargs: t.Any) -> t.Any:
        return self._infer_func(*args, **kwargs)


@inject
def load_runner(
    tag: t.Union[str, Tag],
    *,
    name: t.Optional[str] = None,
    method_name: str = "__call__",
    batch: bool = False,
) -> t.Union[_PicklableModelRunner, _PicklableModelSimpleRunner]:
    """
    Runner represents a unit of serving logic that can be scaled horizontally to
    maximize throughput. :func:`bentoml.picklable_model.load_runner` implements a Runner class that
    wraps the commands that dump and load a pickled object, which optimizes it for the BentoML runtime.

    Args:
        tag (:code:`Union[str, Tag]`):
            Tag of a saved model in BentoML local modelstore..
        method_name:
            Method to call on the pickled object
        batch:
            Determines whether the model supports batching

    Returns:
        :obj:`~bentoml._internal.runner.Runner`: Runner instances for the target :mod:`bentoml.picklable_model` model

    Examples:

    .. code-block:: python

        import bentoml

        runner = bentoml.picklable_model.load_runner("my_model:latest")
        runner.run([[1,2,3,4]])
    """
    if batch:
        return _PicklableModelRunner(
            tag=tag,
            method_name=method_name,
            name=name,
        )
    else:
        return _PicklableModelSimpleRunner(
            tag=tag,
            method_name=method_name,
            name=name,
        )
