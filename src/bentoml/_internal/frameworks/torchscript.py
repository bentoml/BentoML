from __future__ import annotations

import typing as t
import logging
from types import ModuleType
from typing import TYPE_CHECKING

import bentoml
from bentoml import Tag

from ..utils.pkg import get_pkg_version
from ...exceptions import NotFound
from ..models.model import Model
from ..models.model import ModelContext
from ..models.model import PartialKwargsModelOptions as ModelOptions
from .common.pytorch import torch

if TYPE_CHECKING:
    from ..models.model import ModelSignaturesType


logger = logging.getLogger(__name__)
MODULE_NAME = "bentoml.torchscript"
MODEL_FILENAME = "saved_model.pt"
API_VERSION = "v1"


def get(tag_like: str | Tag) -> Model:
    model = bentoml.models.get(tag_like)
    if model.info.module not in (MODULE_NAME, __name__):
        raise NotFound(
            f"Model {model.tag} was saved with module {model.info.module}, not loading with {MODULE_NAME}."
        )
    return model


def load_model(
    bentoml_model: str | Tag | Model,
    device_id: str | None = "cpu",
    *,
    _extra_files: dict[str, t.Any] | None = None,
) -> torch.ScriptModule | tuple[torch.ScriptModule, dict[str, t.Any]]:
    """
    Load a model from BentoML local modelstore with given name.

    Args:
        tag:
            Tag of a saved model in BentoML local modelstore.
        device_id:
            Optional devices to put the given model on. Refer to https://pytorch.org/docs/stable/tensor_attributes.html#torch.torch.device
        _extra_files:
            A dictionary of file names and a empty string. See https://pytorch.org/docs/stable/generated/torch.jit.load.html.

    Returns:
        :obj:`torch.ScriptModule`: an instance of :obj:`torch.ScriptModule` from BentoML modelstore.

    Examples:

    .. code-block:: python

        import bentoml
        lit = bentoml.torchscript.load_model('lit_classifier:latest', device_id="cuda:0")
    """
    if isinstance(bentoml_model, (str, Tag)):
        bentoml_model = get(bentoml_model)

    if bentoml_model.info.module not in (MODULE_NAME, __name__):
        raise NotFound(
            f"Model {bentoml_model.tag} was saved with module {bentoml_model.info.module}, not loading with {MODULE_NAME}."
        )
    weight_file = bentoml_model.path_of(MODEL_FILENAME)

    model: torch.ScriptModule = torch.jit.load(
        weight_file,
        map_location=device_id,
        _extra_files=_extra_files,
    )
    return model


def save_model(
    name: str,
    model: torch.ScriptModule,
    *,
    signatures: ModelSignaturesType | None = None,
    labels: t.Dict[str, str] | None = None,
    custom_objects: t.Dict[str, t.Any] | None = None,
    external_modules: t.List[ModuleType] | None = None,
    metadata: t.Dict[str, t.Any] | None = None,
    _framework_name: str = "torchscript",
    _module_name: str = MODULE_NAME,
    _extra_files: dict[str, t.Any] | None = None,
) -> bentoml.Model:
    """
    Save a model instance to BentoML modelstore.

    Args:
        name (:code:`str`):
            Name for given model instance. This should pass Python identifier check.
        model (`torch.ScriptModule`):
            Instance of model to be saved
        signatures (:code:`dict`, `optional`):
            A dictionary of method names and their corresponding signatures.
        labels (:code:`Dict[str, str]`, `optional`, default to :code:`None`):
            user-defined labels for managing models, e.g. team=nlp, stage=dev
        custom_objects (:code:`Dict[str, Any]]`, `optional`, default to :code:`None`):
            user-defined additional python objects to be saved alongside the model,
            e.g. a tokenizer instance, preprocessor function, model configuration json
        external_modules (:code:`List[ModuleType]`, `optional`, default to :code:`None`):
            user-defined additional python modules to be saved alongside the model or custom objects,
            e.g. a tokenizer module, preprocessor module, model configuration module
        metadata (:code:`Dict[str, Any]`, `optional`, default to :code:`None`):
            Custom metadata for given model.

    Returns:
        :obj:`~bentoml.Tag`: A :obj:`tag` with a format `name:version` where `name` is the user-defined model's name, and a generated `version` by BentoML.

    Examples:

    .. code-block:: python

        import bentoml
        import torch
    """
    if not isinstance(model, (torch.ScriptModule, torch.jit.ScriptModule)):
        raise TypeError(f"Given model ({model}) is not a torch.ScriptModule.")

    if _framework_name == "pytorch_lightning":
        framework_versions = {
            "torch": get_pkg_version("torch"),
            "pytorch_lightning": get_pkg_version("pytorch_lightning"),
        }
    else:
        framework_versions = {"torch": get_pkg_version("torch")}

    context: ModelContext = ModelContext(
        framework_name=_framework_name,
        framework_versions=framework_versions,
    )
    if _extra_files is not None:
        if metadata is None:
            metadata = {}
        metadata["_extra_files"] = [f for f in _extra_files]

    if signatures is None:
        signatures = {"__call__": {"batchable": False}}
        logger.info(
            'Using the default model signature for torchscript (%s) for model "%s".',
            signatures,
            name,
        )

    with bentoml.models.create(
        name,
        module=_module_name,
        api_version=API_VERSION,
        labels=labels,
        signatures=signatures,
        custom_objects=custom_objects,
        external_modules=external_modules,
        options=ModelOptions(),
        context=context,
        metadata=metadata,
    ) as bento_model:
        torch.jit.save(
            model, bento_model.path_of(MODEL_FILENAME), _extra_files=_extra_files
        )
        return bento_model


def get_runnable(bento_model: Model):
    """
    Private API: use :obj:`~bentoml.Model.to_runnable` instead.
    """
    from .common.pytorch import partial_class
    from .common.pytorch import PytorchModelRunnable
    from .common.pytorch import make_pytorch_runnable_method

    partial_kwargs: t.Dict[str, t.Any] = bento_model.info.options.partial_kwargs  # type: ignore
    model_runnable_class = partial_class(
        PytorchModelRunnable,
        bento_model=bento_model,
        loader=load_model,
    )

    for method_name, options in bento_model.info.signatures.items():
        method_partial_kwargs = partial_kwargs.get(method_name)
        model_runnable_class.add_method(
            make_pytorch_runnable_method(method_name, method_partial_kwargs),
            name=method_name,
            batchable=options.batchable,
            batch_dim=options.batch_dim,
            input_spec=options.input_spec,
            output_spec=options.output_spec,
        )
    return model_runnable_class
