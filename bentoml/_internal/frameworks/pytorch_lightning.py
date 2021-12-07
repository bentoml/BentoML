import typing as t
from typing import TYPE_CHECKING

from simple_di import inject
from simple_di import Provide

from .pytorch import _PyTorchRunner as _PyTorchLightningRunner  # type: ignore[reportPrivateUsage] # noqa: LN001

from ...exceptions import MissingDependencyException
from ..types import Tag
from ..models import Model
from ..models import PT_EXT
from ..models import SAVE_NAMESPACE
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
    import torch
    import pytorch_lightning as pl  # noqa: F811
except ImportError:  # pragma: no cover
    raise MissingDependencyException(_PL_IMPORT_ERROR)

try:
    import importlib.metadata as importlib_metadata
except ImportError:
    import importlib_metadata

_torch_version = importlib_metadata.version("torch")
_pl_version = importlib_metadata.version("pytorch_lightning")


@inject
def load(
    tag: t.Union[str, Tag],
    device_id: t.Optional[str] = "cpu",
    model_store: "ModelStore" = Provide[BentoMLContainer.model_store],
) -> "pl.LightningModule":
    """
    Load a model from BentoML local modelstore with given name.

    Args:
        tag (`str`):
            Tag of a saved model in BentoML local modelstore.
        device_id (`str`, `optional`):
            Optional devices to put the given model on. Refers to https://pytorch.org/docs/stable/tensor_attributes.html#torch.torch.device
        model_store (`~bentoml._internal.models.store.ModelStore`, default to `BentoMLContainer.model_store`):
            BentoML modelstore, provided by DI Container.

    Returns:
        an instance of `pl.LightningModule` from BentoML modelstore.

    Examples::
        import bentoml.pytorch_lightning
        booster = bentoml.pytorch_lightning.load(
            'lit_classifier:20201012_DE43A2', device_id="cuda:0")
    """  # noqa: LN001
    bentoml_model = model_store.get(tag)
    weight_file = bentoml_model.path_of(f"{SAVE_NAMESPACE}{PT_EXT}")
    model: "pl.LightningModule" = torch.jit.load(weight_file, map_location=device_id)  # type: ignore[reportPrivateImportUsage] # noqa: LN001
    return model


@inject
def save(
    name: str,
    model: "pl.LightningModule",
    *,
    metadata: t.Optional[t.Dict[str, t.Any]] = None,
    model_store: "ModelStore" = Provide[BentoMLContainer.model_store],
) -> Tag:
    """
    Save a model instance to BentoML modelstore.

    Args:
        name (`str`):
            Name for given model instance. This should pass Python identifier check.
        model (`pl.LightningModule`):
            Instance of model to be saved
        metadata (`t.Optional[t.Dict[str, t.Any]]`, default to `None`):
            Custom metadata for given model.
        model_store (`~bentoml._internal.models.store.ModelStore`, default to `BentoMLContainer.model_store`):
            BentoML modelstore, provided by DI Container.

    Returns:
        tag (`str` with a format `name:version`) where `name` is the defined name user
        set for their models, and version will be generated by BentoML.

    Examples::
        import bentoml.pytorch_lightning
        import pytorch_lightning as pl
        import torch

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
        # example tag: lit_classifier:20201012_DE43A2
    """  # noqa
    context: t.Dict[str, t.Any] = {
        "torch": _torch_version,
        "pytorch_lightning": _pl_version,
    }
    _model = Model.create(
        name,
        module=__name__,
        options=None,
        framework_context=context,
        metadata=metadata,
    )

    weight_file = _model.path_of(f"{SAVE_NAMESPACE}{PT_EXT}")
    torch.jit.save(model.to_torchscript(), weight_file)  # type: ignore[reportUnknownMemberType]

    _model.save(model_store)
    return _model.tag


@inject
def load_runner(
    tag: t.Union[str, Tag],
    *,
    predict_fn_name: str = "__call__",
    device_id: str = "cpu:0",
    partial_kwargs: t.Optional[t.Dict[str, t.Any]] = None,
    resource_quota: t.Union[None, t.Dict[str, t.Any]] = None,
    batch_options: t.Union[None, t.Dict[str, t.Any]] = None,
    model_store: "ModelStore" = Provide[BentoMLContainer.model_store],
) -> "_PyTorchLightningRunner":
    """
        Runner represents a unit of serving logic that can be scaled horizontally to
    maximize throughput. `bentoml.pytorch_lightning.load_runner` implements a Runner class that
    wrap around a statsmodels instance, which optimize it for the BentoML runtime.

    Args:
        tag (`str`):
            Model tag to retrieve model from modelstore
        resource_quota (`t.Dict[str, t.Any]`, default to `None`):
            Dictionary to configure resources allocation for runner.
        predict_fn_name (`str`, default to `__call__`):
            inference function to be used.
        device_id (`t.Union[str, int, t.List[t.Union[str, int]]]`, `optional`, default to `cpu`):
            Optional devices to put the given model on. Refers to https://pytorch.org/docs/stable/tensor_attributes.html#torch.torch.device
        partial_kwargs (`t.Optional[t.Dict[str, t.Any]]`, default to `None`):
            Common kwargs passed to model for this runner
        batch_options (`t.Dict[str, t.Any]`, default to `None`):
            Dictionary to configure batch options for runner in a service context.
        model_store (`~bentoml._internal.models.store.ModelStore`, default to `BentoMLContainer.model_store`):
            BentoML modelstore, provided by DI Container.

    Returns:
        Runner instances for `bentoml.pytorch_lightning` model

    Examples::
        import bentoml.pytorch_lightning
        runner = bentoml.pytorch_lightning.load_runner("lit_classifier:20201012_DE43A2")
        runner.run(pd.DataFrame("/path/to/csv"))
    """  # noqa
    return _PyTorchLightningRunner(
        tag=tag,
        predict_fn_name=predict_fn_name,
        device_id=device_id,
        partial_kwargs=partial_kwargs,
        resource_quota=resource_quota,
        batch_options=batch_options,
        model_store=model_store,
    )
