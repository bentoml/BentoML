from __future__ import annotations

import typing as t
import logging
import functools
from pathlib import Path

import cloudpickle

import bentoml
from bentoml import Tag

from ..models import Model
from ..utils.pkg import get_pkg_version
from ...exceptions import NotFound
from ...exceptions import BentoMLException
from ..models.model import ModelContext
from ..models.model import ModelSignaturesType
from .common.pytorch import torch  # type: ignore
from .common.pytorch import PyTorchTensorContainer

__all__ = ["load_model", "get_runnable", "get", "PyTorchTensorContainer"]

MODULE_NAME = "bentoml.pytorch"
MODEL_FILENAME = "saved_model.pt"

logger = logging.getLogger(__name__)


def get(tag_like: str | Tag) -> Model:
    model = bentoml.models.get(tag_like)
    if model.info.module not in (MODULE_NAME, __name__):
        raise NotFound(
            f"Model {model.tag} was saved with module {model.info.module}, failed loading with {MODULE_NAME}."
        )
    return model


def load_model(
    bentoml_model: str | Tag | Model,
    device_id: t.Optional[str] = "cpu",
) -> torch.nn.Module:
    """
    Load a model from a BentoML Model with given name.

    Args:
        tag (:code:`Union[str, Tag]`):
            Tag of a saved model in BentoML local modelstore.
        device_id (:code:`str`, `optional`, default to :code:`cpu`):
            Optional devices to put the given model on. Refers to `device attributes <https://pytorch.org/docs/stable/tensor_attributes.html#torch.torch.device>`_.

    Returns:
        :obj:`torch.nn.Module`: an instance of :code:`torch.nn.Module` from BentoML modelstore.

    Examples:

    .. code-block:: python

        import bentoml
        model = bentoml.pytorch.load('lit_classifier:latest', device_id="cuda:0")
    """
    if isinstance(bentoml_model, (str, Tag)):
        bentoml_model = get(bentoml_model)

    if bentoml_model.info.module not in (MODULE_NAME, __name__):
        raise BentoMLException(
            f"Model {bentoml_model.tag} was saved with module {bentoml_model.info.module}, failed loading with {MODULE_NAME}."
        )

    weight_file = bentoml_model.path_of(MODEL_FILENAME)
    with Path(weight_file).open("rb") as file:
        model: "torch.nn.Module" = torch.load(file, map_location=device_id)
    return model


def save_model(
    name: str,
    model: "torch.nn.Module",
    *,
    signatures: ModelSignaturesType | None = None,
    labels: t.Dict[str, str] | None = None,
    custom_objects: t.Dict[str, t.Any] | None = None,
    metadata: t.Dict[str, t.Any] | None = None,
) -> Tag:
    """
    Save a model instance to BentoML modelstore.

    Args:
        name (:code:`str`):
            Name for given model instance. This should pass Python identifier check.
        model (:code:`torch.nn.Module`):
            Instance of model to be saved
        labels (:code:`Dict[str, str]`, `optional`, default to :code:`None`):
            user-defined labels for managing models, e.g. team=nlp, stage=dev
        custom_objects (:code:`Dict[str, Any]]`, `optional`, default to :code:`None`):
            user-defined additional python objects to be saved alongside the model,
            e.g. a tokenizer instance, preprocessor function, model configuration json
        metadata (:code:`Dict[str, Any]`, `optional`,  default to :code:`None`):
            Custom metadata for given model.

    Returns:
        :obj:`~bentoml.Tag`: A :obj:`tag` with a format `name:version` where `name` is the user-defined model's name, and a generated `version` by BentoML.

    Examples:

    .. code-block:: python

        import torch
        import bentoml

        class NGramLanguageModeler(nn.Module):

            def __init__(self, vocab_size, embedding_dim, context_size):
                super(NGramLanguageModeler, self).__init__()
                self.embeddings = nn.Embedding(vocab_size, embedding_dim)
                self.linear1 = nn.Linear(context_size * embedding_dim, 128)
                self.linear2 = nn.Linear(128, vocab_size)

            def forward(self, inputs):
                embeds = self.embeddings(inputs).view((1, -1))
                out = F.relu(self.linear1(embeds))
                out = self.linear2(out)
                log_probs = F.log_softmax(out, dim=1)
                return log_probs

        tag = bentoml.pytorch.save("ngrams", NGramLanguageModeler(len(vocab), EMBEDDING_DIM, CONTEXT_SIZE))
        # example tag: ngrams:20201012_DE43A2

    Integration with Torch Hub and BentoML:

    .. code-block:: python

        import torch
        import bentoml

        resnet50 = torch.hub.load("pytorch/vision", "resnet50", pretrained=True)
        ...
        # trained a custom resnet50

        tag = bentoml.pytorch.save("resnet50", resnet50)
    """
    context: ModelContext = ModelContext(
        framework_name="torch",
        framework_versions={"torch": get_pkg_version("torch")},
    )

    if signatures is None:
        raise ValueError("signatures is required for saving a pytorch model")

    with bentoml.models.create(
        name,
        module=MODULE_NAME,
        labels=labels,
        signatures=signatures,
        custom_objects=custom_objects,
        options=None,
        context=context,
        metadata=metadata,
    ) as _model:
        weight_file = _model.path_of(MODEL_FILENAME)
        with open(weight_file, "wb") as file:
            torch.save(model, file, pickle_module=cloudpickle)  # type: ignore

        return _model.tag


def get_runnable(bento_model: Model):
    """
    Private API: use :obj:`~bentoml.Model.to_runnable` instead.
    """
    from .common.pytorch import PytorchModelRunnable
    from .common.pytorch import make_pytorch_runnable_method

    for method_name, options in bento_model.info.signatures.items():
        PytorchModelRunnable.add_method(
            make_pytorch_runnable_method(method_name),
            name=method_name,
            batchable=options.batchable,
            batch_dim=options.batch_dim,
            input_spec=options.input_spec,
            output_spec=options.output_spec,
        )
    return functools.partial(
        PytorchModelRunnable,
        bento_model=bento_model,
        loader=load_model,
    )
