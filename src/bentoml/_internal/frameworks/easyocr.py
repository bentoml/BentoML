from __future__ import annotations

import typing as t
import logging
from types import ModuleType

import cloudpickle

import bentoml

from ..tag import Tag
from ..utils.pkg import get_pkg_version
from ...exceptions import NotFound
from ...exceptions import MissingDependencyException
from ..models.model import Model
from ..models.model import ModelContext
from ..models.model import ModelOptions
from ..models.model import ModelSignature
from .common.pytorch import PyTorchTensorContainer  # noqa # type: ignore

try:
    import easyocr
except ImportError:  # pragma: no cover
    raise MissingDependencyException(
        "'easyocr' is required in order to use module 'bentoml.easyocr'. Install easyocr with 'pip install easyocr'."
    )

if t.TYPE_CHECKING:
    from ..models.model import ModelSignaturesType

    ListStr = list[str]
else:
    ListStr = list


__all__ = ["load_model", "save_model", "get_runnable", "get"]

MODULE_NAME = "bentoml.easyocr"
API_VERSION = "v1"
MODEL_FILENAME = "saved_model.pkl"


logger = logging.getLogger(__name__)


def get(tag_like: str | Tag) -> Model:
    """
    Get the BentoML model with the given tag.

    Args:
        tag_like: The tag of the model to retrieve from the model store.

    Returns:
        :obj:`~bentoml.Model`: A BentoML :obj:`~bentoml.Model` with the matching tag.

    Example:

    .. code-block:: python

       import bentoml
       # target model must be from the BentoML model store
       model = bentoml.easyocr.get("en_reader:latest")
    """
    model = bentoml.models.get(tag_like)
    if model.info.module not in (MODULE_NAME, __name__):
        raise NotFound(
            f"Model {model.tag} was saved with module {model.info.module}, not loading with {MODULE_NAME}."
        )
    return model


def load_model(bento_model: str | Tag | Model) -> easyocr.Reader:
    """
    Load the EasyOCR model from BentoML local model store with given name.

    Args:
        bento_model: Either the tag of the model to get from the store,
                     or a BentoML :class:`~bentoml.Model` instance to load the
                     model from.

    Returns:
        ``easyocr.Reader``: The EasyOCR model from the model store.

    Example:

    .. code-block:: python

        import bentoml
        reader = bentoml.easyocr.load_model('en_reader:latest')
    """
    if not isinstance(bento_model, Model):
        bento_model = get(bento_model)

    if bento_model.info.module not in (MODULE_NAME, __name__):
        raise NotFound(
            f"Model {bento_model.tag} was saved with module {bento_model.info.module}, not loading with {MODULE_NAME}."
        )
    with open(bento_model.path_of(MODEL_FILENAME), "rb") as f:
        return cloudpickle.load(f)


def save_model(
    name: Tag | str,
    reader: easyocr.Reader,
    *,
    signatures: ModelSignaturesType | None = None,
    labels: dict[str, str] | None = None,
    custom_objects: dict[str, t.Any] | None = None,
    external_modules: t.List[ModuleType] | None = None,
    metadata: dict[str, t.Any] | None = None,
) -> bentoml.Model:
    """
    Save a model instance to BentoML modelstore.

    Args:
        name: Name for given model instance. This should pass Python identifier check.
        reader: The EasyOCR model to be saved. Currently only supports pre-trained models from easyocr.
                Custom models are not yet supported.
        signatures: Methods to expose for running inference on the target model. Signatures are used for creating :obj:`~bentoml.Runner` instances when serving model with :obj:`~bentoml.Service`
        labels: User-defined labels for managing models, e.g. ``team=nlp``, ``stage=dev``.
        custom_objects: Custom objects to be saved with the model. An example is ``{"my-normalizer": normalizer}``.
                        Custom objects are currently serialized with cloudpickle, but this implementation is subject to change.
        external_modules: user-defined additional python modules to be saved alongside the model or custom objects,
                          e.g. a tokenizer module, preprocessor module, model configuration module
        metadata: Custom metadata for given model.

    Returns:
        :obj:`~bentoml.Tag`: A :obj:`tag` with a format ``name:version`` where ``name`` is the user-defined model's name, and a generated ``version``.

    Examples:

    .. code-block:: python

        import bentoml
        import easyocr

        reader = easyocr.Reader(['en'])
        bento_model = bentoml.easyocr.save_model('en_reader', reader)
    """  # noqa
    context = ModelContext(
        framework_name="easyocr",
        framework_versions={"easyocr": get_pkg_version("easyocr")},
    )

    if signatures is None:
        signatures = {
            k: {"batchable": False}
            for k in ("detect", "readtext", "readtextlang", "recognize")
        }
        signatures["readtext_batched"] = {"batchable": True}
        logger.info(
            'Using the default model signature for Transformers (%s) for model "%s".',
            signatures,
            name,
        )

    with bentoml.models.create(
        name,
        module=MODULE_NAME,
        api_version=API_VERSION,
        labels=labels,
        context=context,
        options=ModelOptions(),
        signatures=signatures,
        custom_objects=custom_objects,
        external_modules=external_modules,
        metadata=metadata,
    ) as bento_model:
        with open(bento_model.path_of(MODEL_FILENAME), "wb") as f:
            cloudpickle.dump(reader, f)
        return bento_model


def get_runnable(bento_model: bentoml.Model) -> type[bentoml.Runnable]:
    """
    Private API: use :obj:`~bentoml.Model.to_runnable` instead.
    """

    class EasyOCRRunnable(bentoml.Runnable):
        SUPPORTED_RESOURCES = ("nvidia.com/gpu", "cpu")
        SUPPORTS_CPU_MULTI_THREADING = True

        def __init__(self):
            super().__init__()

            self.model = load_model(bento_model)

            self.predict_fns: dict[str, t.Callable[..., t.Any]] = {}
            for method_name in bento_model.info.signatures:
                self.predict_fns[method_name] = getattr(self.model, method_name)

    def add_runnable_method(method_name: str, options: ModelSignature):
        def _run(self: EasyOCRRunnable, *args: t.Any, **kwargs: t.Any) -> t.Any:
            return self.predict_fns[method_name](*args, **kwargs)

        EasyOCRRunnable.add_method(
            _run,
            name=method_name,
            batchable=options.batchable,
            batch_dim=options.batch_dim,
            input_spec=options.input_spec,
            output_spec=options.output_spec,
        )

    for method_name, options in bento_model.info.signatures.items():
        add_runnable_method(method_name, options)

    return EasyOCRRunnable
