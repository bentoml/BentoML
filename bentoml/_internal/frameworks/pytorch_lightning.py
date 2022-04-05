import typing as t
from typing import TYPE_CHECKING

import torch
from simple_di import inject
from simple_di import Provide

import bentoml
from bentoml import Tag
from bentoml.exceptions import BentoMLException
from bentoml.exceptions import MissingDependencyException

from ..models import PT_EXT
from ..models import SAVE_NAMESPACE
from ..utils.pkg import get_pkg_version
from .common.pytorch import torch
from .common.pytorch import BasePyTorchRunner
from ..configuration.containers import BentoMLContainer

_PL_IMPORT_ERROR = f"""\
`pytorch_lightning` and `torch` is required in order to use module `{__name__}`\n
Refers to https://pytorch.org/get-started/locally/ to setup PyTorch correctly.
Then run `pip install pytorch_lightning`
"""

if TYPE_CHECKING:
    import pytorch_lightning as pl

    from ..models import ModelStore

try:
    import pytorch_lightning as pl  # noqa: F811
except ImportError:  # pragma: no cover
    raise MissingDependencyException(_PL_IMPORT_ERROR)

MODULE_NAME = "bentoml.pytorch_lightning"


@inject
def load(
    tag: Tag,
    device_id: t.Optional[str] = "cpu",
    model_store: "ModelStore" = Provide[BentoMLContainer.model_store],
) -> "pl.LightningModule":
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
    """  # noqa: LN001
    bentoml_model = model_store.get(tag)
    if bentoml_model.info.module not in (MODULE_NAME, __name__):
        raise BentoMLException(
            f"Model {tag} was saved with module {bentoml_model.info.module}, failed loading with {MODULE_NAME}."
        )
    weight_file = bentoml_model.path_of(f"{SAVE_NAMESPACE}{PT_EXT}")
    model_format = bentoml_model.info.context.get("model_format")
    # backward compatibility
    if not model_format:
        model_format = "pytorch_lightning:v1"

    if model_format == "pytorch_lightning:v1":
        model: "pl.LightningModule" = torch.jit.load(weight_file, map_location=device_id)  # type: ignore[reportPrivateImportUsage] # noqa: LN001
    else:
        raise BentoMLException(f"Unknown model format {model_format}")

    return model


def save(
    name: str,
    model: "pl.LightningModule",
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
    """  # noqa
    context: t.Dict[str, t.Any] = {
        "framework_name": "torch",
        "pip_dependencies": [
            f"torch=={get_pkg_version('torch')}",
            f"pytorch_lightning=={get_pkg_version('pytorch_lightning')}",
        ],
    }

    with bentoml.models.create(
        name,
        module=MODULE_NAME,
        options=None,
        context=context,
        labels=labels,
        custom_objects=custom_objects,
        metadata=metadata,
    ) as _model:

        weight_file = _model.path_of(f"{SAVE_NAMESPACE}{PT_EXT}")
        _model.info.context["model_format"] = "pytorch_lightning:v1"
        torch.jit.save(model.to_torchscript(), weight_file)  # type: ignore[reportUnknownMemberType]

        return _model.tag


class _PyTorchLightningRunner(BasePyTorchRunner):
    def _load_model(self):
        return load(self._tag, device_id=self._device_id, model_store=self.model_store)


@inject
def load_runner(
    tag: t.Union[str, Tag],
    *,
    predict_fn_name: str = "__call__",
    partial_kwargs: t.Optional[t.Dict[str, t.Any]] = None,
    name: t.Optional[str] = None,
) -> "_PyTorchLightningRunner":
    """
    Runner represents a unit of serving logic that can be scaled horizontally to
    maximize throughput. :func:`bentoml.pytorch_lightning.load_runner` implements a Runner class that
    wrap around a :obj:`pl.LightningModule` instance, which optimize it for the BentoML runtime.

    Args:
        tag (:code:`Union[str, Tag]`):
            Tag of a saved model in BentoML local modelstore.
        predict_fn_name (:code:`str`, default to :code:`__call__`):
            inference function to be used.
        partial_kwargs (:code:`Dict[str, Any]`, `optional`,  default to :code:`None`):
            Common kwargs passed to model for this runner

    Returns:
        :obj:`~bentoml._internal.runner.Runner`: Runner instances for :mod:`bentoml.pytorch_lightning` model

    Examples:

    .. code-block:: python

        import bentoml.pytorch_lightning
        runner = bentoml.pytorch_lightning.load_runner("lit_classifier:20201012_DE43A2")
        runner.run(pd.DataFrame("/path/to/csv"))
    """
    return _PyTorchLightningRunner(
        tag=tag,
        predict_fn_name=predict_fn_name,
        name=name,
        partial_kwargs=partial_kwargs,
    )
