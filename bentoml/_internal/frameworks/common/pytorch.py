import pickle
import typing as t
import logging
import functools
import contextlib
from abc import ABC
from abc import abstractmethod
from typing import TYPE_CHECKING

from simple_di import inject

from ...tag import Tag
from ...types import LazyType
from ...utils.pkg import get_pkg_version
from .model_runner import BaseModelRunner
from ....exceptions import MissingDependencyException
from ...runner.utils import Params
from ...runner.container import Payload
from ...runner.container import DataContainer
from ...runner.container import DataContainerRegistry

try:
    import torch
    import torch.nn.parallel as parallel
except ImportError:  # pragma: no cover
    raise MissingDependencyException(
        """\
        torch is required in order to use module `bentoml.pytorch`,
        `bentoml.torchscript` and `bentoml.pytorch_lightning`.
        Instruction: Refers to https://pytorch.org/get-started/locally/
        to setup PyTorch correctly.  """  # noqa
    )

if TYPE_CHECKING:
    import pytorch_lightning as pl

    from ... import external_typing as ext

    ModelType = t.Union[torch.nn.Module, torch.jit.ScriptModule, pl.LightningModule]

logger = logging.getLogger(__name__)


class BasePyTorchRunner(BaseModelRunner, ABC):
    def __init__(
        self,
        tag: t.Union[str, Tag],
        predict_fn_name: str,
        partial_kwargs: t.Optional[t.Dict[str, t.Any]],
        name: t.Optional[str] = None,
    ):
        super().__init__(tag=tag, name=name)

        self._predict_fn_name = predict_fn_name
        self._partial_kwargs = partial_kwargs or dict()

        self._predict_fn: t.Callable[..., torch.Tensor]
        self._no_grad_context: t.Optional[contextlib.ExitStack] = None
        self._model: t.Optional["ModelType"] = None

    @property
    def _device_id(self):
        if self._on_gpu:
            return "cuda"
        else:
            return "cpu"

    @property
    def _num_threads(self) -> int:
        if self._on_gpu:
            return 1
        return max(round(self.resource_quota.cpu), 1)

    @property
    def num_replica(self) -> int:
        if self._on_gpu:
            return torch.cuda.device_count()
        return 1

    def _configure(self) -> None:
        if self._on_gpu:
            torch.set_default_tensor_type("torch.cuda.FloatTensor")
        else:
            torch.set_num_threads(self._num_threads)
            torch.set_default_tensor_type("torch.FloatTensor")

    @property
    def _on_gpu(self) -> bool:
        if self.resource_quota.on_gpu:
            if torch.cuda.is_available():
                return True
            else:
                logger.warning(
                    "GPU is not available, but GPU resource is requested. "
                    "Falling back to CPU."
                )
        return False

    @abstractmethod
    def _load_model(self):
        raise NotImplementedError

    def _setup(self) -> None:
        self._no_grad_context = contextlib.ExitStack()
        self._no_grad_context.enter_context(torch.no_grad())
        if get_pkg_version("torch").startswith("1.9"):
            # inference mode is required for PyTorch version 1.9.*
            self._no_grad_context.enter_context(torch.inference_mode())

        self._configure()
        model = self._load_model()
        model.eval()
        if self._on_gpu:
            self._model = parallel.DataParallel(model)
            torch.cuda.empty_cache()
        else:
            self._model = model
        raw_predict_fn = getattr(self._model, self._predict_fn_name)
        self._predict_fn = functools.partial(raw_predict_fn, **self._partial_kwargs)

    def _shutdown(self) -> None:
        if self._no_grad_context is not None:
            self._no_grad_context.close()
            self._no_grad_context = None

    def _run_batch(
        self,
        *args: t.Union["ext.NpNDArray", torch.Tensor],
        **kwargs: t.Union["ext.NpNDArray", torch.Tensor],
    ) -> torch.Tensor:

        params = Params[t.Union["ext.NpNDArray", torch.Tensor]](*args, **kwargs)

        def _mapping(item: t.Union["ext.NpNDArray", torch.Tensor]) -> torch.Tensor:
            if LazyType["ext.NpNDArray"]("numpy.ndarray").isinstance(item):
                item = torch.Tensor(item, device=self._device_id)
            else:
                item = item.to(self._device_id)
            return item

        params = params.map(_mapping)
        res = self._predict_fn(*params.args, **kwargs)
        return res


class PyTorchTensorContainer(DataContainer[torch.Tensor, torch.Tensor]):
    @classmethod
    def singles_to_batch(cls, singles, batch_axis=0):
        return torch.stack(singles, dim=batch_axis)

    @classmethod
    def batch_to_singles(cls, batch, batch_axis=0):
        return [
            torch.squeeze(tensor, dim=batch_axis)
            for tensor in torch.split(batch, 1, dim=batch_axis)
        ]

    @classmethod
    @inject
    def single_to_payload(
        cls,
        single,
    ) -> Payload:
        return cls.create_payload(
            pickle.dumps(single),
            {"plasma": False},
        )

    @classmethod
    @inject
    def payload_to_single(
        cls,
        payload: Payload,
    ):
        return pickle.loads(payload.data)

    batch_to_payload = single_to_payload
    payload_to_batch = payload_to_single


DataContainerRegistry.register_container(
    LazyType("torch", "Tensor"),
    LazyType("torch", "Tensor"),
    PyTorchTensorContainer,
)
