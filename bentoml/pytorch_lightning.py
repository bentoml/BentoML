import functools
import typing as t
from pathlib import Path

from simple_di import Provide, WrappedCallable
from simple_di import inject as _inject

from ._internal.configuration.containers import BentoMLContainer
from ._internal.models import SAVE_NAMESPACE
from .exceptions import MissingDependencyException

_PL_IMPORT_ERROR = f"""\
`pytorch_lightning` and `torch` is required in order to use module `{__name__}`\n
Refers to https://pytorch.org/get-started/locally/ to setup PyTorch correctly.
Then run `pip install pytorch_lightning`
"""
_PT_EXTENSION = ".pt"

if t.TYPE_CHECKING:  # pragma: no cover
    # pylint: disable=unused-import
    from ._internal.models.store import ModelStore

try:
    import pytorch_lightning as pl
    import torch

    from bentoml.pytorch import _PyTorchRunner as _PyTorchLightningRunner
except ImportError:  # pragma: no cover
    raise MissingDependencyException(_PL_IMPORT_ERROR)


inject: t.Callable[[WrappedCallable], WrappedCallable] = functools.partial(
    _inject, squeeze_none=False
)


@inject
def load(
    tag: str,
    device_id: t.Optional[str] = "cpu",
    model_store: "ModelStore" = Provide[BentoMLContainer.model_store],
) -> "pl.LightningModule":
    model_info = model_store.get(tag)
    weight_file = Path(model_info.path, f"{SAVE_NAMESPACE}{_PT_EXTENSION}")
    _load: t.Callable[[str], "pl.LightningModule"] = functools.partial(
        torch.jit.load, map_location=device_id
    )
    return _load(str(weight_file))


@inject
def save(
    name: str,
    model: "pl.LightningModule",
    *,
    metadata: t.Union[None, t.Dict[str, t.Any]] = None,
    model_store: "ModelStore" = Provide[BentoMLContainer.model_store],
) -> str:
    context = dict(torch=torch.__version__, pytorch_lightning=pl.__version__)
    with model_store.register(
        name,
        module=__name__,
        options=None,
        framework_context=context,
        metadata=metadata,
    ) as ctx:
        weight_file = Path(ctx.path, f"{SAVE_NAMESPACE}{_PT_EXTENSION}")
        torch.jit.save(model.to_torchscript(), str(weight_file))
        return ctx.tag


@inject
def load_runner(
    tag: str,
    *,
    resource_quota: t.Union[None, t.Dict[str, t.Any]] = None,
    batch_options: t.Union[None, t.Dict[str, t.Any]] = None,
    model_store: "ModelStore" = Provide[BentoMLContainer.model_store],
) -> "_PyTorchLightningRunner":
    _runner: t.Callable[[str], "_PyTorchLightningRunner"] = functools.partial(
        _PyTorchLightningRunner,
        predict_fn_name="__call__",
        device_id="cpu",
        resource_quota=resource_quota,
        batch_options=batch_options,
        model_store=model_store,
    )
    return _runner(tag)
