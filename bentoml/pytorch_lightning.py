import functools
import typing as t
from pathlib import Path

from simple_di import Provide, WrappedCallable
from simple_di import inject as _inject

from ._internal.configuration.containers import BentoMLContainer
from ._internal.models import SAVE_NAMESPACE
from ._internal.runner import Runner
from .exceptions import MissingDependencyException

_PL_IMPORT_ERROR = f"""\
`pytorch_lightning` and `torch` is required in order to use module `{__name__}`\n
Refers to https://pytorch.org/get-started/locally/ to setup PyTorch correctly.
Then run `pip install pytorch_lightning`
"""
_PT_EXTENSION = ".pt"

if t.TYPE_CHECKING:  # pragma: no cover
    # pylint: disable=unused-import
    import pytorch_lightning as pl
    import torch
    import torch.nn as nn

    from ._internal.models.store import ModelStore

try:
    import pytorch_lightning as pl
    import torch
    import torch.nn as nn
except ImportError:  # pragma: no cover
    raise MissingDependencyException(_PL_IMPORT_ERROR)

inject: t.Callable[[WrappedCallable], WrappedCallable] = functools.partial(
    _inject, squeeze_none=False
)


@_inject
def load(
    tag: str,
    model_store: "ModelStore" = Provide[BentoMLContainer.model_store],
) -> "pl.LightningModule":
    model_info = model_store.get(tag)
    weight_file = Path(model_info.path, f"{SAVE_NAMESPACE}{_PT_EXTENSION}")
    return torch.jit.load(str(weight_file))


@inject
def save(
    name: str,
    model: "pl.LightningModule",
    *,
    metadata: t.Union[None, t.Dict[str, t.Any]] = None,
    model_store: "ModelStore" = Provide[BentoMLContainer.model_store],
):
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


class _PyTorchLightningRunner(Runner):
    def __init__(
        self,
        tag: str,
        resource_quota: t.Dict[str, t.Any],
        batch_options: t.Dict[str, t.Any],
        model_store: "ModelStore" = Provide[BentoMLContainer.model_store],
    ):
        super().__init__(tag, resource_quota, batch_options)

    @property
    def required_models(self) -> t.List[str]:
        ...

    @property
    def num_concurrency_per_replica(self) -> int:
        ...

    @property
    def num_replica(self) -> int:
        ...

    def _setup(self, **kwargs) -> None:
        ...

    def _run_batch(self, *args, **kwargs) -> t.Any:
        ...


@inject
def load_runner(
    tag: str,
    *,
    resource_quota: t.Union[None, t.Dict[str, t.Any]] = None,
    batch_options: t.Union[None, t.Dict[str, t.Any]] = None,
    model_store: "ModelStore" = Provide[BentoMLContainer.model_store],
):
    return _PyTorchLightningRunner(
        tag=tag,
        resource_quota=resource_quota,
        batch_options=batch_options,
        model_store=model_store,
    )
