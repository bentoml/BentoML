from __future__ import annotations

import typing as t
import logging
from typing import TYPE_CHECKING

import bentoml
from bentoml import Tag
from bentoml.models import Model
from bentoml.models import ModelContext
from bentoml.exceptions import NotFound
from bentoml.exceptions import BentoMLException
from bentoml.exceptions import MissingDependencyException

from ..models import PKL_EXT
from ..models import SAVE_NAMESPACE
from ..models.model import ModelSignature

if TYPE_CHECKING:
    from .. import external_typing as ext

    ModelType = t.Any

MODULE_NAME = "bentoml.picklable_model"
API_VERSION = "v1"

try:
    import cloudpickle  # type: ignore
except ImportError:  # pragma: no cover
    raise MissingDependencyException(
        f"""cloudpickle is required in order to use the module `{MODULE_NAME}`, install
         cloudpickle with `pip install cloudpickle`.
        """
    )


logger = logging.getLogger(__name__)


def get(tag_like: str | Tag) -> Model:
    model = bentoml.models.get(tag_like)
    if model.info.module not in (MODULE_NAME, __name__):
        raise NotFound(
            f"Model {model.tag} was saved with module {model.info.module}, failed loading with {MODULE_NAME}."
        )
    return model


def load_model(bento_model: str | Tag | Model) -> ModelType:
    """
    Load the picklable model with the given tag from the local BentoML model store.

    Args:
        bento_model (``str`` ``|`` :obj:`~bentoml.Tag` ``|`` :obj:`~bentoml.Model`):
            Either the tag of the model to get from the store, or a BentoML `~bentoml.Model`
            instance to load the model from.
        ...
    Returns:
        ``object``
            The picklable model loaded from the model store or BentoML :obj:`~bentoml.Model`.
    Example:
    .. code-block:: python
        import bentoml

        picklable_model = bentoml.picklable_model.load_model('my_model:latest')
    """  # noqa
    if not isinstance(bento_model, Model):
        bento_model = get(bento_model)

    if bento_model.info.module not in (MODULE_NAME, __name__):
        raise BentoMLException(
            f"Model {bento_model.tag} was saved with module {bento_model.info.module}, failed loading with {MODULE_NAME}."
        )
    model_file = bento_model.path_of(f"{SAVE_NAMESPACE}{PKL_EXT}")

    with open(model_file, "rb") as f:
        # The protocol version used is detected automatically, so we do not
        # have to specify it.
        return cloudpickle.load(f)


def save_model(
    name: str,
    model: ModelType,
    *,
    signatures: dict[str, ModelSignature] | None = None,
    labels: t.Dict[str, str] | None = None,
    custom_objects: t.Dict[str, t.Any] | None = None,
    metadata: t.Dict[str, t.Any] | None = None,
) -> Tag:
    """
    Save a model instance to BentoML modelstore.

    Args:
        name (:code:`str`):
            Name for given model instance. This should pass Python identifier check.
        model (:code:`Union[BaseEstimator, Pipeline]`):
            Instance of model to be saved.
        signatures (:code: `Dict[str, ModelSignatureDict]`)
            Methods to expose for running inference on the target model. Signatures are
             used for creating Runner instances when serving model with bentoml.Service
        labels (:code:`Dict[str, str]`, `optional`, default to :code:`None`):
            user-defined labels for managing models, e.g. team=nlp, stage=dev
        custom_objects (:code:`Dict[str, Any]]`, `optional`, default to :code:`None`):
            user-defined additional python objects to be saved alongside the model,
             e.g. a tokenizer instance, preprocessor function, model configuration json
        metadata (:code:`Dict[str, Any]`, `optional`,  default to :code:`None`):
            Custom metadata for given model.

    Returns:
        :obj:`~bentoml.Tag`: A :obj:`tag` with a format `name:version` where `name` is
        the user-defined model's name, and a generated `version`.

    Examples:

    .. code-block:: python

        import bentoml

        tag = bentoml.picklable.save_model('picklable_pyobj', model)

        # load the model back:
        loaded = bentoml.picklable.load_model("picklable_pyobj:latest")
        # or:
        loaded = bentoml.picklable.load_model(tag)
    """  # noqa
    context = ModelContext(
        framework_name="cloudpickle",
        framework_versions={"cloudpickle": cloudpickle.__version__},
    )

    if signatures is None:
        logger.info(
            'Using default model signature `{"__call__": {"batchable": False}}` for picklable model'
        )
        signatures = {"__call__": ModelSignature(batchable=False)}

    with bentoml.models.create(
        name,
        module=MODULE_NAME,
        api_version=API_VERSION,
        labels=labels,
        custom_objects=custom_objects,
        metadata=metadata,
        context=context,
        signatures=signatures,
    ) as _model:

        with open(_model.path_of(f"{SAVE_NAMESPACE}{PKL_EXT}"), "wb") as f:
            cloudpickle.dump(model, f)

        return _model.tag


def get_runnable(bento_model: Model):
    """
    Private API: use :obj:`~bentoml.Model.to_runnable` instead.
    """

    class PicklableRunnable(bentoml.Runnable):
        SUPPORT_NVIDIA_GPU = False  # type: ignore
        SUPPORT_CPU_MULTI_THREADING = False  # type: ignore

        def __init__(self):
            super().__init__()
            self.model = load_model(bento_model)

    def _get_run(method_name: str):
        def _run(
            self: PicklableRunnable,
            *args: ext.NpNDArray | ext.PdDataFrame,
            **kwargs: ext.NpNDArray | ext.PdDataFrame,
        ) -> ext.NpNDArray:
            assert isinstance(method_name, str), repr(method_name)
            return getattr(self.model, method_name)(*args, **kwargs)

        return _run

    for method_name, options in bento_model.info.signatures.items():
        assert isinstance(method_name, str), repr(method_name)
        PicklableRunnable.add_method(
            _get_run(method_name),
            name=method_name,
            batchable=options.batchable,
            batch_dim=options.batch_dim,
            input_spec=options.input_spec,
            output_spec=options.output_spec,
        )

    return PicklableRunnable
