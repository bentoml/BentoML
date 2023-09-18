from __future__ import annotations

import pickle
import typing as t
import logging
import functools
import itertools
from typing import TYPE_CHECKING

from simple_di import inject
from simple_di import Provide

import bentoml

from ...types import LazyType
from ....exceptions import MissingDependencyException
from ...models.model import Model
from ...runner.utils import Params
from ...runner.container import Payload
from ...runner.container import DataContainer
from ...runner.container import DataContainerRegistry
from ...configuration.containers import BentoMLContainer

try:
    import torch
except ImportError:  # pragma: no cover
    raise MissingDependencyException(
        "'torch' is required in order to use module 'bentoml.pytorch', 'bentoml.torchscript' or 'bentoml.pytorch_lightning'. Install torch with 'pip install torch'. For more information, refer to https://pytorch.org/get-started/locally/"
    )

if TYPE_CHECKING:
    import pytorch_lightning as pl

    from ... import external_typing as ext

    ModelType = torch.nn.Module | torch.ScriptModule | pl.LightningModule
    T = t.TypeVar("T")

logger = logging.getLogger(__name__)

if hasattr(torch, "inference_mode"):  # pytorch>=1.9
    inference_mode_ctx = torch.inference_mode
else:
    inference_mode_ctx = torch.no_grad


def partial_class(
    cls: type[PytorchModelRunnable], *args: t.Any, **kwargs: t.Any
) -> type[PytorchModelRunnable]:
    class NewClass(cls):
        def __init__(self, *inner_args: t.Any, **inner_kwargs: t.Any) -> None:
            functools.partial(cls.__init__, *args, **kwargs)(
                self, *inner_args, **inner_kwargs
            )

    return NewClass


class PytorchModelRunnable(bentoml.Runnable):
    SUPPORTED_RESOURCES = ("nvidia.com/gpu", "cpu")
    SUPPORTS_CPU_MULTI_THREADING = True

    def __init__(
        self,
        bento_model: Model,
        loader: t.Callable[..., torch.nn.Module],
    ):
        super().__init__()
        # if torch.cuda.device_count():
        if torch.cuda.is_available():
            self.device_id = "cuda"
            # by default, torch.FloatTensor will be used on CPU.
            torch.set_default_tensor_type("torch.cuda.FloatTensor")
        else:
            self.device_id = "cpu"
        self.model: ModelType = loader(bento_model, device_id=self.device_id)
        # We want to turn off dropout and batchnorm when running inference.
        self.model.train(False)


def make_pytorch_runnable_method(
    method_name: str,
    partial_kwargs: dict[str, t.Any] | None = None,
) -> t.Callable[..., torch.Tensor]:
    if partial_kwargs is None:
        partial_kwargs = {}

    def _run(
        self: PytorchModelRunnable,
        *args: ext.PdDataFrame | ext.NpNDArray | torch.Tensor,
        **kwargs: ext.PdDataFrame | ext.NpNDArray | torch.Tensor,
    ) -> torch.Tensor:
        params = Params(*args, **kwargs)

        def _mapping(item: T) -> torch.Tensor | T:
            if LazyType["ext.NpNDArray"]("numpy.ndarray").isinstance(item):
                return torch.Tensor(item, device=self.device_id)
            if LazyType["ext.PdDataFrame"]("pandas.DataFrame").isinstance(item):
                return torch.Tensor(item.to_numpy(), device=self.device_id)
            if LazyType["torch.Tensor"]("torch.Tensor").isinstance(item):
                return item.to(self.device_id)
            else:
                return item

        with inference_mode_ctx():
            params = params.map(_mapping)
            return getattr(self.model, method_name)(
                *params.args,
                **dict(partial_kwargs, **params.kwargs),
            )

    return _run


class PyTorchTensorContainer(DataContainer[torch.Tensor, torch.Tensor]):
    @classmethod
    def batches_to_batch(
        cls,
        batches: t.Sequence[torch.Tensor],
        batch_dim: int = 0,
    ) -> t.Tuple[torch.Tensor, list[int]]:
        batch = torch.cat(tuple(batches), dim=batch_dim)
        indices = list(
            itertools.accumulate(subbatch.shape[batch_dim] for subbatch in batches)
        )
        indices = [0] + indices
        return batch, indices

    @classmethod
    def batch_to_batches(
        cls,
        batch: torch.Tensor,
        indices: t.Sequence[int],
        batch_dim: int = 0,
    ) -> t.List[torch.Tensor]:
        sizes = [indices[i] - indices[i - 1] for i in range(1, len(indices))]
        output: list[torch.Tensor] = torch.split(batch, sizes, dim=batch_dim)
        return output

    @classmethod
    @inject
    def to_payload(  # pylint: disable=arguments-differ
        cls,
        batch: torch.Tensor,
        batch_dim: int = 0,
        plasma_db: "ext.PlasmaClient" | None = Provide[BentoMLContainer.plasma_db],
    ) -> Payload:
        batch = batch.cpu().numpy()
        if plasma_db:
            return cls.create_payload(
                plasma_db.put(batch).binary(),
                batch_size=batch.shape[batch_dim],
                meta={"plasma": True},
            )

        return cls.create_payload(
            pickle.dumps(batch),
            batch_size=batch.shape[batch_dim],
            meta={"plasma": False},
        )

    @classmethod
    @inject
    def from_payload(  # pylint: disable=arguments-differ
        cls,
        payload: Payload,
        plasma_db: "ext.PlasmaClient" | None = Provide[BentoMLContainer.plasma_db],
    ) -> torch.Tensor:
        if payload.meta.get("plasma"):
            import pyarrow.plasma as plasma

            assert plasma_db
            ret = plasma_db.get(plasma.ObjectID(payload.data))

        else:
            ret = pickle.loads(payload.data)
        return torch.from_numpy(ret).requires_grad_(False)

    @classmethod
    @inject
    def batch_to_payloads(  # pylint: disable=arguments-differ
        cls,
        batch: torch.Tensor,
        indices: t.Sequence[int],
        batch_dim: int = 0,
        plasma_db: "ext.PlasmaClient" | None = Provide[BentoMLContainer.plasma_db],
    ) -> t.List[Payload]:
        batches = cls.batch_to_batches(batch, indices, batch_dim)
        payloads = [cls.to_payload(i, batch_dim=batch_dim) for i in batches]
        return payloads

    @classmethod
    @inject
    def from_batch_payloads(  # pylint: disable=arguments-differ
        cls,
        payloads: t.Sequence[Payload],
        batch_dim: int = 0,
        plasma_db: "ext.PlasmaClient" | None = Provide[BentoMLContainer.plasma_db],
    ) -> t.Tuple[torch.Tensor, list[int]]:
        batches = [cls.from_payload(payload, plasma_db) for payload in payloads]
        return cls.batches_to_batch(batches, batch_dim)


DataContainerRegistry.register_container(
    LazyType("torch", "Tensor"),
    LazyType("torch", "Tensor"),
    PyTorchTensorContainer,
)
