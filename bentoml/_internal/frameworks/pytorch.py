import typing as t
import logging
from typing import TYPE_CHECKING
from pathlib import Path

import cloudpickle
from simple_di import inject
from simple_di import Provide

from bentoml import Tag
from bentoml import Model

from ..models import PT_EXT
from ..models import SAVE_NAMESPACE
from ..utils.pkg import get_pkg_version
from ...exceptions import BentoMLException
from ...exceptions import MissingDependencyException
from .common.pytorch import torch
from .common.pytorch import BasePyTorchRunner
from .common.pytorch import PyTorchTensorContainer  # pylint: disable=unused-import
from ..configuration.containers import BentoMLContainer

if TYPE_CHECKING:
    from ..models import ModelStore

MODULE_NAME = "bentoml.pytorch"

logger = logging.getLogger(__name__)


@inject
def load(
    tag: t.Union[Tag, str],
    device_id: t.Optional[str] = "cpu",
    model_store: "ModelStore" = Provide[BentoMLContainer.model_store],
) -> "torch.nn.Module":
    """
    Load a model from BentoML local modelstore with given name.

    Args:
        tag (:code:`Union[str, Tag]`):
            Tag of a saved model in BentoML local modelstore.
        device_id (:code:`str`, `optional`, default to :code:`cpu`):
            Optional devices to put the given model on. Refers to `device attributes <https://pytorch.org/docs/stable/tensor_attributes.html#torch.torch.device>`_.
        model_store (:mod:`~bentoml._internal.models.store.ModelStore`, default to :mod:`BentoMLContainer.model_store`):
            BentoML modelstore, provided by DI Container.

    Returns:
        :obj:`torch.nn.Module`: an instance of :code:`torch.nn.Module` from BentoML modelstore.

    Examples:

    .. code-block:: python

        import bentoml
        model = bentoml.pytorch.load('lit_classifier:latest', device_id="cuda:0")
    """  # noqa
    bentoml_model = model_store.get(tag)
    if bentoml_model.info.module not in (MODULE_NAME, __name__):
        raise BentoMLException(
            f"Model {tag} was saved with module {bentoml_model.info.module}, failed loading with {MODULE_NAME}."
        )
    weight_file = bentoml_model.path_of(f"{SAVE_NAMESPACE}{PT_EXT}")
    model_format = bentoml_model.info.context.get("model_format")
    # backward compatibility
    if not model_format:
        model_format = "cloudpickle:v1"

    if model_format == "cloudpickle:v1":
        with Path(weight_file).open("rb") as file:
            model: "torch.nn.Module" = cloudpickle.load(file).to(device_id)
    elif model_format == "torch.save:v1":
        with Path(weight_file).open("rb") as file:
            model: "torch.nn.Module" = torch.load(file, map_location=device_id)
    else:
        raise BentoMLException(f"Unknown model format {model_format}")

    return model


@inject
def save(
    name: str,
    model: "torch.nn.Module",
    *,
    metadata: t.Union[None, t.Dict[str, t.Any]] = None,
    model_store: "ModelStore" = Provide[BentoMLContainer.model_store],
) -> Tag:
    """
    Save a model instance to BentoML modelstore.

    Args:
        name (:code:`str`):
            Name for given model instance. This should pass Python identifier check.
        model (:code:`torch.nn.Module`):
            Instance of model to be saved
        metadata (:code:`Dict[str, Any]`, `optional`,  default to :code:`None`):
            Custom metadata for given model.
        model_store (:mod:`~bentoml._internal.models.store.ModelStore`, default to :mod:`BentoMLContainer.model_store`):
            BentoML modelstore, provided by DI Container.

    Returns:
        :obj:`~bentoml._internal.types.Tag`: A :obj:`tag` with a format `name:version` where `name` is the user-defined model's name, and a generated `version` by BentoML.

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
    """  # noqa
    context: t.Dict[str, t.Any] = {
        "framework_name": "torch",
        "pip_dependencies": [f"torch=={get_pkg_version('torch')}"],
    }
    _model = Model.create(
        name,
        module=MODULE_NAME,
        options=None,
        context=context,
        metadata=metadata,
    )
    weight_file = _model.path_of(f"{SAVE_NAMESPACE}{PT_EXT}")
    _model.info.context["model_format"] = "torch.save:v1"
    with open(weight_file, "wb") as file:
        torch.save(model, file, pickle_module=cloudpickle)

    _model.save(model_store)
    return _model.tag


class _PyTorchRunner(BasePyTorchRunner):
    def _load_model(self):
        return load(self._tag, device_id=self._device_id)


def load_runner(
    tag: t.Union[str, Tag],
    *,
    predict_fn_name: str = "__call__",
    partial_kwargs: t.Optional[t.Dict[str, t.Any]] = None,
    name: t.Optional[str] = None,
) -> "_PyTorchRunner":
    """
        Runner represents a unit of serving logic that can be scaled horizontally to
        maximize throughput. `bentoml.pytorch.load_runner` implements a Runner class that
        wrap around a pytorch instance, which optimize it for the BentoML runtime.

    Args:
        tag (:code:`Union[str, Tag]`):
            Tag of a saved model in BentoML local modelstore.
        predict_fn_name (:code:`str`, default to :code:`__call__`):
            inference function to be used.
        partial_kwargs (:code:`Dict[str, Any]`, `optional`,  default to :code:`None`):
            Common kwargs passed to model for this runner

    Returns:
        :obj:`~bentoml._internal.runner.Runner`: Runner instances for :mod:`bentoml.pytorch` model

    Examples:

    .. code-block:: python

        import bentoml
        import pandas as pd

        runner = bentoml.pytorch.load_runner("ngrams:latest")
        runner.run(pd.DataFrame("/path/to/csv"))
    """
    return _PyTorchRunner(
        tag=tag,
        predict_fn_name=predict_fn_name,
        partial_kwargs=partial_kwargs,
        name=name,
    )
