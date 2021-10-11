import functools
import typing as t
import zipfile
from pathlib import Path

import cloudpickle
from simple_di import Provide, WrappedCallable
from simple_di import inject as _inject

from ._internal.configuration.containers import BentoMLContainer
from ._internal.models import SAVE_NAMESPACE
from ._internal.runner import Runner
from .exceptions import MissingDependencyException

_PT_EXTENSION = ".pt"

if t.TYPE_CHECKING:  # pragma: no cover
    # pylint: disable=unused-import
    import torch
    import torch.nn as nn

    from ._internal.models.store import ModelStore

try:
    import torch
    import torch.nn as nn
except ImportError:  # pragma: no cover
    raise MissingDependencyException(
        "torch is required in order "
        "to use module `bentoml.pytorch`. "
        "Refers to https://pytorch.org/get-started/locally/ to setup PyTorch correctly."
    )

inject: t.Callable[[WrappedCallable], WrappedCallable] = functools.partial(
    _inject, squeeze_none=False
)


@inject
def load(
    tag: str,
    model_store: "ModelStore" = Provide[BentoMLContainer.model_store],
) -> t.Union["torch.nn.Module", "torch.jit.ScriptModule"]:
    model_info = model_store.get(tag)
    weight_file = Path(model_info.path, f"{SAVE_NAMESPACE}{_PT_EXTENSION}")
    # TorchScript Models are saved as zip files
    # This also includes pl.LightningModule
    if zipfile.is_zipfile(str(weight_file)):
        return torch.jit.load(str(weight_file))
    else:
        with weight_file.open("rb") as file:
            return cloudpickle.load(file)


@inject
def save(
    name: str,
    model: t.Union["torch.nn.Module", "torch.jit.ScriptModule", "pl.LightningModule"],
    *,
    metadata: t.Union[None, t.Dict[str, t.Any]] = None,
    model_store: "ModelStore" = Provide[BentoMLContainer.model_store],
):
    context = dict(torch=torch.__version__)
    with model_store.register(
        name,
        module=__name__,
        options=None,
        framework_context=context,
        metadata=metadata,
    ) as ctx:
        weight_file = Path(ctx.path, f"{SAVE_NAMESPACE}{_PT_EXTENSION}")
        if isinstance(model, torch.jit.ScriptModule):
            torch.jit.save(model, str(weight_file))
        else:
            with weight_file.open("wb") as file:
                cloudpickle.dump(model, file)


class _PyTorchRunner(Runner):
    def __init__(
        self,
        tag: str,
        resource_quota: t.Dict[str, t.Any],
        batch_options: t.Dict[str, t.Any],
        model_store: "ModelStore" = Provide[BentoMLContainer.model_store],
    ):
        super().__init__(tag, resource_quota, batch_options)
        ...

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
    return _PyTorchRunner(
        tag=tag,
        resource_quota=resource_quota,
        batch_options=batch_options,
        model_store=model_store,
    )
