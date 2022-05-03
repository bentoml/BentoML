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
from ..configuration.containers import BentoMLContainer

if TYPE_CHECKING:
    from ..models import ModelStore

MODULE_NAME = "bentoml.picklable"


@inject
def load_model(
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
        return cloudpickle.load(f)


def save_model(
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
        tag_info = bentoml.picklable.save_model("test_pickle_model", model_to_save)
        loaded_model = bentoml.picklable.load_model("test_pickle_model:latest")

    Using saved pickable model in Service via Runner:

    .. code-block:: python
        runner = bentoml.picklable.get("test_pickle_model").to_runner()
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
