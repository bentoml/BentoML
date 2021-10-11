import functools
import re
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
_ModelType = t.TypeVar(
    "_ModelType", bound=t.Union["torch.nn.Module", "torch.jit.ScriptModule"]
)

if t.TYPE_CHECKING:  # pragma: no cover
    # pylint: disable=unused-import
    import numpy as np

    from ._internal.models.store import ModelStore

try:
    import torch
    import torch.nn.parallel as parallel
except ImportError:  # pragma: no cover
    raise MissingDependencyException(
        "torch is required in order "
        "to use module `bentoml.pytorch`. "
        "Refers to https://pytorch.org/get-started/locally/ to setup PyTorch correctly."
    )

infer_mode_compat = torch.__version__.startswith("1.9")

inject: t.Callable[[WrappedCallable], WrappedCallable] = functools.partial(
    _inject, squeeze_none=False
)


def _is_gpu_enabled() -> bool:  # pragma: no cover
    return torch.cuda.is_available()


def _clean_name(name: str) -> str:  # pragma: no cover
    return re.sub(r"\W|^(?=\d)-", "_", name)


@inject
def load(
    tag: str,
    device_id: t.Optional[str] = "cpu",
    model_store: "ModelStore" = Provide[BentoMLContainer.model_store],
) -> _ModelType:
    model_info = model_store.get(tag)
    weight_file = Path(model_info.path, f"{SAVE_NAMESPACE}{_PT_EXTENSION}")
    # TorchScript Models are saved as zip files
    if zipfile.is_zipfile(str(weight_file)):
        _load: t.Callable[[str], _ModelType] = functools.partial(
            torch.jit.load, map_location=device_id
        )
        return _load(str(weight_file))
    else:
        with weight_file.open("rb") as file:
            _cload: t.Callable[[t.BinaryIO], _ModelType] = functools.partial(
                cloudpickle.load
            )
            return _cload(file)


@inject
def save(
    name: str,
    model: _ModelType,
    *,
    metadata: t.Union[None, t.Dict[str, t.Any]] = None,
    model_store: "ModelStore" = Provide[BentoMLContainer.model_store],
) -> str:
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
        return ctx.tag


@inject
def import_from_torch_hub(
    repo: str,
    model: str,
    *args: str,
    model_store: "ModelStore" = Provide[BentoMLContainer.model_store],
    **kwargs: str,
) -> str:
    with model_store.register(
        _clean_name(repo),
        module=__name__,
        options=None,
        framework_context=dict(torch=torch.__version__),
    ) as ctx:
        torch.hub.set_dir(ctx.path)
        torch.hub.load(
            repo,
            model,
            *args,
            **kwargs,
        )
        return ctx.tag


class _PyTorchRunner(Runner):
    @inject
    def __init__(
        self,
        tag: str,
        predict_fn_name: str,
        resource_quota: t.Dict[str, t.Any],
        batch_options: t.Dict[str, t.Any],
        device_id: t.Union[str, int, t.List[t.Union[str, int]]],
        model_store: "ModelStore" = Provide[BentoMLContainer.model_store],
    ):
        super().__init__(tag, resource_quota, batch_options)
        self._model_store = model_store
        self._predict_fn_name = predict_fn_name
        self._configure_torch(device_id)

    def _configure_torch(
        self, device_id: t.Union[str, int, t.List[t.Union[str, int]]]
    ) -> None:
        if isinstance(device_id, list):
            self._devices = list(torch.device(dev) for dev in device_id)
        else:
            self._devices = [torch.device(device_id)]
        if _is_gpu_enabled():
            torch.set_default_tensor_type("torch.cuda.FloatTensor")
        else:
            torch.set_default_tensor_type("torch.FloatTensor")
        torch.set_num_threads(self.num_concurrency_per_replica)
        torch.set_num_interop_threads(self.num_concurrency_per_replica)

    @property
    def required_models(self) -> t.List[str]:
        return [self._model_store.get(self.name).tag]

    @property
    def num_concurrency_per_replica(self) -> int:
        # TODO(aarnphm): use resource_quota.gpus instead
        if _is_gpu_enabled():
            return 1
        return int(round(self.resource_quota.cpu))

    @property
    def num_replica(self) -> int:
        # TODO(aarnphm): use resource_quota.gpus instead
        if _is_gpu_enabled():
            return torch.cuda.device_count()
        return 1

    # pylint: disable=arguments-differ,attribute-defined-outside-init
    @torch.no_grad()
    def _setup(self) -> None:  # type: ignore[override]
        self._model = parallel.DistributedDataParallel(
            load(self.name, model_store=self._model_store, device_id=None),
            device_ids=self._devices,
        )
        if self.resource_quota.on_gpu:
            torch.cuda.empty_cache()
        self._predict_fn = getattr(self._model, self._predict_fn_name)

    # pylint: disable=arguments-differ,attribute-defined-outside-init
    @torch.no_grad()
    def _run_batch(self, *inputs: "np.ndarray", **kwargs: str) -> t.Any:
        if infer_mode_compat:
            with torch.inference_mode():
                return self._predict_fn(*inputs, **kwargs)
        return self._predict_fn(*inputs, **kwargs)


@inject
def load_runner(
    tag: str,
    *,
    predict_fn_name: str = "__call__",
    resource_quota: t.Union[None, t.Dict[str, t.Any]] = None,
    batch_options: t.Union[None, t.Dict[str, t.Any]] = None,
    device_id: t.Union[str, int, t.List[t.Union[str, int]]] = "cpu",
    model_store: "ModelStore" = Provide[BentoMLContainer.model_store],
) -> "_PyTorchRunner":
    _runner: t.Callable[[str], "_PyTorchRunner"] = functools.partial(
        _PyTorchRunner,
        predict_fn_name=predict_fn_name,
        resource_quota=resource_quota,
        batch_options=batch_options,
        device_id=device_id,
        model_store=model_store,
    )
    return _runner(tag)
