import typing as t
from typing import TYPE_CHECKING

from simple_di import inject
from simple_di import Provide

from bentoml import Tag
from bentoml import Model
from bentoml.exceptions import BentoMLException
from bentoml.exceptions import MissingDependencyException

from ..models import PT_EXT
from ..models import SAVE_NAMESPACE
from .pytorch import _PyTorchRunner as _PyTorchLightningRunner  # type: ignore[reportPrivateUsage] # noqa: LN001
from ..utils.pkg import get_pkg_version
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
        name (:code:`str`):
            Name for given model instance. This should pass Python identifier check.
        model (`pl.LightningModule`):
            Instance of model to be saved
        metadata (:code:`Dict[str, Any]`, `optional`, default to :code:`None`):
            Custom metadata for given model.
        model_store (:mod:`~bentoml._internal.models.store.ModelStore`, default to :mod:`BentoMLContainer.model_store`):
            BentoML modelstore, provided by DI Container.

    Returns:
        :obj:`~bentoml._internal.types.Tag`: A :obj:`tag` with a format `name:version` where `name` is the user-defined model's name, and a generated `version` by BentoML.

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
    _model = Model.create(
        name,
        module=MODULE_NAME,
        options=None,
        context=context,
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
    name: t.Optional[str] = None,
    resource_quota: t.Union[None, t.Dict[str, t.Any]] = None,
    batch_options: t.Union[None, t.Dict[str, t.Any]] = None,
    model_store: "ModelStore" = Provide[BentoMLContainer.model_store],
) -> "_PyTorchLightningRunner":
    """
    Runner represents a unit of serving logic that can be scaled horizontally to
    maximize throughput. :func:`bentoml.pytorch_lightning.load_runner` implements a Runner class that
    wrap around a :obj:`pl.LightningModule` instance, which optimize it for the BentoML runtime.

    Args:
        tag (:code:`Union[str, Tag]`):
            Tag of a saved model in BentoML local modelstore.
        resource_quota (:code:`Dict[str, Any]`, default to :code:`None`):
            Dictionary to configure resources allocation for runner.
        predict_fn_name (:code:`str`, default to :code:`__call__`):
            inference function to be used.
        device_id (:code:`Union[str, int, List[Union[str, int]]]`, `optional`, default to :code:`cpu`):
            Optional devices to put the given model on. Refers to `Tensor Attributes Docs <https://pytorch.org/docs/stable/tensor_attributes.html#torch.torch.device>`_
            for more information.
        partial_kwargs (:code:`Dict[str, Any]`, `optional`,  default to :code:`None`):
            Common kwargs passed to model for this runner
        batch_options (:code:`Dict[str, Any]`, default to :code:`None`):
            Dictionary to configure batch options for runner in a service context.
        model_store (:mod:`~bentoml._internal.models.store.ModelStore`, default to :mod:`BentoMLContainer.model_store`):
            BentoML modelstore, provided by DI Container.

    Returns:
        :obj:`~bentoml._internal.runner.Runner`: Runner instances for :mod:`bentoml.pytorch_lightning` model

    Examples:

    .. code-block:: python

        import bentoml.pytorch_lightning
        runner = bentoml.pytorch_lightning.load_runner("lit_classifier:20201012_DE43A2")
        runner.run(pd.DataFrame("/path/to/csv"))
    """  # noqa
    tag = Tag.from_taglike(tag)
    if name is None:
        name = tag.name
    return _PyTorchLightningRunner(
        tag=tag,
        predict_fn_name=predict_fn_name,
        name=name,
        device_id=device_id,
        partial_kwargs=partial_kwargs,
        resource_quota=resource_quota,
        batch_options=batch_options,
        model_store=model_store,
    )
