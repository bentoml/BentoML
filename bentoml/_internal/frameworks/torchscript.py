from __future__ import annotations

import typing as t
import functools

import torch

import bentoml
from bentoml import Tag

from ..utils.pkg import get_pkg_version
from ...exceptions import NotFound
from ...exceptions import BentoMLException
from ...exceptions import MissingDependencyException
from ..models.model import Model
from ..models.model import ModelContext
from ..models.model import ModelSignaturesType
from .common.pytorch import torch

_PL_IMPORT_ERROR = f"""\
`torch` is required in order to use module `{__name__}`\n
Refers to https://pytorch.org/get-started/locally/ to setup PyTorch correctly.
Then run `pip install pytorch_lightning`
"""


try:
    import torch
except ImportError:  # pragma: no cover
    raise MissingDependencyException(_PL_IMPORT_ERROR)

MODULE_NAME = "bentoml.pytorch_lightning"
MODEL_FILENAME = "savd_model.pt"


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
) -> torch.ScriptModule:
    """
    Load a model from BentoML local modelstore with given name.

    Args:
        tag (:code:`Union[str, Tag]`):
            Tag of a saved model in BentoML local modelstore.
        device_id (:code:`str`, `optional`):
            Optional devices to put the given model on. Refers to https://pytorch.org/docs/stable/tensor_attributes.html#torch.torch.device
        model_store (:mod:`~bentoml._internal.models.store.ModelStore`, default to :mod:`BentoMLContainer.model_store`):
            BentoML modelstore, provided by DI Container.

    Returns:
        :obj:`pl.LightningModule`: an instance of :obj:`pl.LightningModule` from BentoML modelstore.

    Examples:

    .. code-block:: python

        import bentoml
        lit = bentoml.pytorch_lightning.load('lit_classifier:latest', device_id="cuda:0")
    """
    if isinstance(bentoml_model, (str, Tag)):
        bentoml_model = get(bentoml_model)

    if bentoml_model.info.module not in (MODULE_NAME, __name__):
        raise BentoMLException(
            f"Model {bentoml_model} was saved with module {bentoml_model.info.module}, failed loading with {MODULE_NAME}."
        )
    weight_file = bentoml_model.path_of(MODEL_FILENAME)
    model: torch.ScriptModule = torch.jit.load(weight_file, map_location=device_id)  # type: ignore[reportPrivateImportUsage]
    return model


def save_model(
    name: str,
    model: torch.ScriptModule,
    *,
    signatures: ModelSignaturesType | None = None,
    labels: t.Dict[str, str] | None = None,
    custom_objects: t.Dict[str, t.Any] | None = None,
    metadata: t.Dict[str, t.Any] | None = None,
    _include_pytorch_lightning_version: bool = False,
) -> Tag:
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

        tag = bentoml.pytorch_lightning.save("lit_classifier", LitClassifier())
    """

    if _include_pytorch_lightning_version:
        framework_versions = {
            "torch": get_pkg_version("torch"),
            "pytorch_lightning": get_pkg_version("pytorch_lightning"),
        }
    else:
        framework_versions = {"torch": get_pkg_version("torch")}

    context: ModelContext = ModelContext(
        framework_name="torchscript",
        framework_versions=framework_versions,
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
        torch.jit.save(model, weight_file)  # type: ignore
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
