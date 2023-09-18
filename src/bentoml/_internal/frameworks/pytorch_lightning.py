from __future__ import annotations

import typing as t
from types import ModuleType
from typing import TYPE_CHECKING

import bentoml
from bentoml import Tag

from .torchscript import save_model as script_save_model
from .torchscript import MODEL_FILENAME
from ...exceptions import NotFound
from ...exceptions import MissingDependencyException
from ..models.model import Model
from .common.pytorch import torch

if TYPE_CHECKING:
    from ..models.model import ModelSignaturesType

try:
    import pytorch_lightning as pl
except ImportError:  # pragma: no cover
    raise MissingDependencyException(
        "'pytorch_lightning' is required in order to use module 'bentoml.pytorch_lightning', install python-lightning with: 'pip install pytorch-lightning'"
    )

MODULE_NAME = "bentoml.pytorch_lightning"


__all__ = ["save_model", "load_model", "get_runnable", "get"]


def get(tag_like: str | Tag) -> Model:
    model = bentoml.models.get(tag_like)
    if model.info.module not in (MODULE_NAME, __name__):
        raise NotFound(
            f"Model {model.tag} was saved with module {model.info.module}, not loading with {MODULE_NAME}."
        )
    return model


def load_model(
    bentoml_model: str | Tag | Model,
    device_id: t.Optional[str] = "cpu",
) -> torch.ScriptModule:
    """
    Load a model from BentoML local modelstore with given name.

    Args:
        tag (:code:`Union[str, Tag]`):
            Tag of a saved model in BentoML local modelstore.
        device_id (:code:`str`, `optional`):
            Optional devices to put the given model on. Refer to https://pytorch.org/docs/stable/tensor_attributes.html#torch.torch.device
        model_store (:mod:`~bentoml._internal.models.store.ModelStore`, default to :mod:`BentoMLContainer.model_store`):
            BentoML modelstore, provided by DI Container.

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
    model: torch.ScriptModule = torch.jit.load(weight_file, map_location=device_id)
    return model


def save_model(
    name: Tag | str,
    model: pl.LightningModule,
    *,
    signatures: ModelSignaturesType | None = None,
    labels: t.Dict[str, str] | None = None,
    custom_objects: t.Dict[str, t.Any] | None = None,
    external_modules: t.List[ModuleType] | None = None,
    metadata: t.Dict[str, t.Any] | None = None,
) -> bentoml.Model:
    """
    Save a model instance to BentoML modelstore.

    Args:
        name (:code:`str`):
            Name for given model instance. This should pass Python identifier check.
        model (`pl.LightningModule`):
            Instance of model to be saved
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
        model_store (:mod:`~bentoml._internal.models.store.ModelStore`, default to :mod:`BentoMLContainer.model_store`):
            BentoML modelstore, provided by DI Container.

    Returns:
        :obj:`~bentoml.Tag`: A :obj:`tag` with a format `name:version` where `name` is the user-defined model's name, and a generated `version` by BentoML.

    Examples:

    .. code-block:: python

        import bentoml
        import torch
        import pytorch_lightning as pl

        class LitClassifier(pl.LightningModule):

            def __init__(self, hidden_dim: int = 128, learning_rate: float = 0.0001):
                super().__init__()
                self.save_hyperparameters()

                self.l1 = torch.nn.Linear(28 * 28, self.hparams.hidden_dim)
                self.l2 = torch.nn.Linear(self.hparams.hidden_dim, 10)

            def forward(self, x):
                x = x.view(x.size(0), -1)
                x = torch.relu(self.l1(x))
                x = torch.relu(self.l2(x))
                return x

            def training_step(self, batch, batch_idx):
                x, y = batch
                y_hat = self(x)
                loss = F.cross_entropy(y_hat, y)
                return loss

            def validation_step(self, batch, batch_idx):
                x, y = batch
                y_hat = self(x)
                loss = F.cross_entropy(y_hat, y)
                self.log("valid_loss", loss)

            def test_step(self, batch, batch_idx):
                x, y = batch
                y_hat = self(x)
                loss = F.cross_entropy(y_hat, y)
                self.log("test_loss", loss)

            def configure_optimizers(self):
                return torch.optim.Adam(self.parameters(), lr=self.hparams.learning_rate)

        tag = bentoml.pytorch_lightning.save_model("lit_classifier", LitClassifier())
    """
    if not isinstance(model, pl.LightningModule):
        raise TypeError(
            f"'model' must be an instance of 'pl.LightningModule', got {type(model)} instead."
        )

    script_module = model.to_torchscript()

    assert not isinstance(
        script_module, dict
    ), "Saving a dict of pytorch_lightning Module into one BentoModel is not supported"

    return script_save_model(
        name,
        script_module,
        signatures=signatures,
        labels=labels,
        custom_objects=custom_objects,
        external_modules=external_modules,
        metadata=metadata,
        _framework_name="pytorch_lightning",
        _module_name=MODULE_NAME,
    )


def get_runnable(bento_model: Model):
    """
    Private API: use :obj:`~bentoml.Model.to_runnable` instead.
    """
    from .common.pytorch import partial_class
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
    return partial_class(
        PytorchModelRunnable,
        bento_model=bento_model,
        loader=load_model,
    )
