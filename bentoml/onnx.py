import logging
import os
import shutil
import typing as t
from typing import TYPE_CHECKING

from simple_di import Provide, inject

from ._internal.configuration.containers import BentoMLContainer
from ._internal.models import SAVE_NAMESPACE
from ._internal.runner import Runner
from ._internal.runner.utils import Params, _get_gpu_memory
from ._internal.types import PathType
from ._internal.utils import LazyLoader
from .exceptions import BentoMLException, MissingDependencyException

SUPPORTED_ONNX_BACKEND: t.List[str] = ["onnxruntime", "onnxruntime-gpu"]
ONNX_EXT: str = ".onnx"

if TYPE_CHECKING:  # pragma: no cover
    import pandas as pd
    from _internal.models.store import ModelInfo, ModelStore, StoreCtx

try:
    import numpy as np
    import onnx
    import onnxruntime as ort
except ImportError:  # pragma: no cover
    raise MissingDependencyException(
        """\
`onnx` is required in order to use the module `bentoml.onnx`, do `pip install onnx`.
For more information, refers to https://onnx.ai/get-started.html
`onnxruntime` is also required by `bentoml.onnx`. Refer to https://onnxruntime.ai/ for
more information.
        """
    )

pd = LazyLoader("pd", globals(), "pandas")  # noqa: F811

_ProviderType = t.TypeVar(
    "_ProviderType", bound=t.List[t.Union[str, t.Tuple[str, t.Dict[str, t.Any]]]]
)

logger = logging.getLogger(__name__)


# helper methods
def _yield_providers(
    iterable: t.Sequence[t.Any],
) -> t.Generator[str, None, None]:  # pragma: no cover
    if isinstance(iterable, tuple):
        yield iterable[0]
    elif isinstance(iterable, str):
        yield iterable
    else:
        yield from iterable


def flatten_list(lst: t.List[t.Any]) -> t.List[str]:  # pragma: no cover
    return [k for i in lst for k in _yield_providers(i)]


def _get_model_info(
    tag: str,
    model_store: "ModelStore",
) -> t.Tuple["ModelInfo", str]:
    model_info = model_store.get(tag)
    if model_info.module != __name__:
        raise BentoMLException(
            f"Model {tag} was saved with module {model_info.module}, failed loading "
            f"with {__name__}."
        )
    model_file = os.path.join(model_info.path, f"{SAVE_NAMESPACE}{ONNX_EXT}")
    return model_info, model_file


@inject
def load(
    tag: str,
    backend: t.Optional[str] = "onnxruntime",
    providers: t.Optional[_ProviderType] = None,
    session_options: t.Optional["ort.SessionOptions"] = None,
    model_store: "ModelStore" = Provide[BentoMLContainer.model_store],
) -> "ort.InferenceSession":
    """
    Load a model from BentoML local modelstore with given name.

    Args:
        tag (`str`):
            Tag of a saved model in BentoML local modelstore.
        backend (`str`, `optional`, default to `onnxruntime`):
            Different backend runtime supported by ONNX. Currently only accepted `onnxruntime`
             and `onnxruntime-gpu`.
        providers (`List[Union[str, t.Tuple[str, Dict[str, Any]]`, `optional`, default to `None`):
            Different providers provided by users. By default BentoML will use `onnxruntime.get_available_providers()`
             when loading a model.
        session_options (`onnxruntime.SessionOptions`, `optional`, default to `None`):
            SessionOptions per usecase. If not specified, then default to None.
        model_store (`~bentoml._internal.models.store.ModelStore`, default to `BentoMLContainer.model_store`):
            BentoML modelstore, provided by DI Container.

    Returns:
        an instance of Onnx model from BentoML modelstore.

    Examples::
    """  # noqa
    _, model_file = _get_model_info(tag, model_store)

    if backend not in SUPPORTED_ONNX_BACKEND:
        raise BentoMLException(
            f"'{backend}' runtime is currently not supported for ONNXModel"
        )
    if providers:
        if not all(i in ort.get_all_providers() for i in flatten_list(providers)):
            raise BentoMLException(f"'{providers}' cannot be parsed by `onnxruntime`")
    else:
        providers = ort.get_available_providers()

    return ort.InferenceSession(
        model_file, sess_options=session_options, providers=providers
    )


@inject
def save(
    name: str,
    model: t.Union[onnx.ModelProto, PathType],
    *,
    metadata: t.Union[None, t.Dict[str, t.Any]] = None,
    model_store: "ModelStore" = Provide[BentoMLContainer.model_store],
) -> str:
    """
    Save a model instance to BentoML modelstore.

    Args:
        name (`str`):
            Name for given model instance. This should pass Python identifier check.
        model:
            Instance of model to be saved
        metadata (`t.Optional[t.Dict[str, t.Any]]`, default to `None`):
            Custom metadata for given model.
        model_store (`~bentoml._internal.models.store.ModelStore`, default to `BentoMLContainer.model_store`):
            BentoML modelstore, provided by DI Container.

    Returns:
        tag (`str` with a format `name:version`) where `name` is the defined name user
        set for their models, and version will be generated by BentoML.

    Examples::
    """  # noqa
    context = {"onnx": onnx.__version__, "onnxruntime": ort.__version__}
    with model_store.register(
        name,
        module=__name__,
        metadata=metadata,
        framework_context=context,
    ) as ctx:  # type: StoreCtx
        if isinstance(model, onnx.ModelProto):
            onnx.save_model(
                model, os.path.join(ctx.path, f"{SAVE_NAMESPACE}{ONNX_EXT}")
            )
        else:
            shutil.copyfile(
                model, os.path.join(ctx.path, f"{SAVE_NAMESPACE}{ONNX_EXT}")
            )
        _tag = ctx.tag  # type: str
        return _tag


class _ONNXRunner(Runner):
    @inject
    def __init__(
        self,
        tag: str,
        backend: str,
        gpu_device_id: int,
        disable_copy_in_default_stream: bool,
        providers: t.Optional[_ProviderType],
        session_options: t.Optional["ort.SessionOptions"],
        resource_quota: t.Optional[t.Dict[str, t.Any]],
        batch_options: t.Optional[t.Dict[str, t.Any]],
        model_store: "ModelStore",
    ):
        if gpu_device_id != -1:
            resource_quota = dict() if not resource_quota else resource_quota
            if "gpus" not in resource_quota:
                resource_quota["gpus"] = gpu_device_id

        super().__init__(tag, resource_quota, batch_options)
        self._model_info, self._model_file = _get_model_info(tag, model_store)
        self._model_store = model_store
        self._backend = backend

        if backend not in SUPPORTED_ONNX_BACKEND:
            raise BentoMLException(
                f"'{backend}' runtime is currently not supported for ONNXModel"
            )
        if providers is not None:
            if not all(i in ort.get_all_providers() for i in flatten_list(providers)):
                raise BentoMLException(
                    f"'{providers}' cannot be parsed by `onnxruntime`"
                )
        else:
            providers = self._get_default_providers(
                gpu_device_id, disable_copy_in_default_stream
            )
        self._providers = providers
        self._session_options = self._get_default_session_options(session_options)

    @staticmethod
    def _get_default_providers(
        gpu_device_id: int, disable_copy_in_default_stream: bool
    ) -> _ProviderType:
        if gpu_device_id != -1:
            _, free = _get_gpu_memory(gpu_device_id)
            gpu_ = {
                "device_id": gpu_device_id,
                "arena_extend_strategy": "kNextPowerOfTwo",
                "gpu_mem_limit": free,
                "cudnn_conv_algo_search": "EXHAUSTIVE",
                "do_copy_in_default_stream": True,
            }
            if disable_copy_in_default_stream:
                logger.warning(
                    "`disable_copy_in_default_stream=True` will set `do_copy_in_default_stream=False`."
                    " There are race conditions and possibly better performance."
                )
                gpu_["do_copy_in_default_stream"] = False
            providers = [
                ("CUDAExecutionProvider", gpu_),
                "CPUExecutionProvider",
            ]
        else:
            providers = ort.get_available_providers()
        return providers  # type: ignore[return-value]

    def _get_default_session_options(
        self, session_options: t.Optional["ort.SessionOptions"]
    ) -> "ort.SessionOptions":
        if session_options is not None:
            return session_options
        _session_options = ort.SessionOptions()
        _session_options.execution_mode = ort.ExecutionMode.ORT_PARALLEL
        _session_options.intra_op_num_threads = self.num_concurrency_per_replica
        _session_options.inter_op_num_threads = self.num_concurrency_per_replica
        _session_options.graph_optimization_level = (
            ort.GraphOptimizationLevel.ORT_ENABLE_EXTENDED
        )
        return _session_options

    @property
    def required_models(self) -> t.List[str]:
        return [self._model_info.tag]

    @property
    def num_concurrency_per_replica(self) -> int:
        # TODO: support GPU threads
        if self.resource_quota.on_gpu:
            return 1
        return int(round(self.resource_quota.cpu))

    @property
    def num_replica(self) -> int:
        if self.resource_quota.on_gpu:
            return len(self.resource_quota.gpus)
        return 1

    # pylint: disable=arguments-differ,attribute-defined-outside-init
    def _setup(self) -> None:  # type: ignore[override]
        self._model = load(
            self._model_info.tag,
            backend=self._backend,
            providers=self._providers,
            session_options=self._session_options,
            model_store=self._model_store,
        )
        self._infer_func = getattr(self._model, "run")

    def _run_batch(
        self,
        *args: t.Union["np.ndarray", "pd.DataFrame"],
        **kwargs: t.Any,
    ) -> t.Any:
        params = Params[t.Union["np.ndarray", "pd.DataFrame"]](*args, **kwargs)
        if isinstance(params.sample, np.ndarray):
            params = params.map(lambda i: i.astype(np.float32))
        elif isinstance(params.sample, pd.DataFrame):
            params = params.map(lambda i: i.to_numpy())
        else:
            raise TypeError(
                f"`_run_batch` of {self.__class__.__name__} only takes "
                "`numpy.ndarray` or `pd.DataFrame` as input parameters"
            )
        input_names = {
            i.name: val for i, val in zip(self._model.get_inputs(), params.args)
        }
        output_names = [_.name for _ in self._model.get_outputs()]
        return self._infer_func(output_names, input_names, **params.kwargs)


@inject
def load_runner(
    tag: str,
    *,
    backend: str = "onnxruntime",
    gpu_device_id: int = -1,
    disable_copy_in_default_stream: bool = False,
    providers: t.Optional[_ProviderType] = None,
    session_options: t.Optional["ort.SessionOptions"] = None,
    resource_quota: t.Optional[t.Dict[str, t.Any]] = None,
    batch_options: t.Optional[t.Dict[str, t.Any]] = None,
    model_store: "ModelStore" = Provide[BentoMLContainer.model_store],
) -> "_ONNXRunner":
    """
    Runner represents a unit of serving logic that can be scaled horizontally to
    maximize throughput. `bentoml.onnx.load_runner` implements a Runner class that
    wrap around an ONNX model, which optimize it for the BentoML runtime.

    Args:
        tag (`str`):
            Model tag to retrieve model from modelstore
        gpu_device_id (`int`, `optional`, default to `-1`):
            GPU device ID. Currently only support CUDA.
        disable_copy_in_default_stream (`bool`, `optional`, default to `False`):
            Whether to do copies in the default stream or use separate streams. Refers to
             https://onnxruntime.ai/docs/execution-providers/CUDA-ExecutionProvider.html#do_copy_in_default_stream
        backend (`str`, `optional`, default to `onnxruntime`):
            Different backend runtime supported by ONNX. Currently only accepted `onnxruntime`
             and `onnxruntime-gpu`.
        providers (`List[Union[str, t.Tuple[str, Dict[str, Any]]`, `optional`, default to `None`):
            Different providers provided by users. By default BentoML will use `CPUExecutionProvider` when
             loading a model
        session_options (`onnxruntime.SessionOptions`, `optional`, default to `None`):
            SessionOptions per usecase. If not specified, then default to None.
        resource_quota (`t.Dict[str, t.Any]`, default to `None`):
            Dictionary to configure resources allocation for runner.
        batch_options (`t.Dict[str, t.Any]`, default to `None`):
            Dictionary to configure batch options for runner in a service context.
        model_store (`~bentoml._internal.models.store.ModelStore`, default to `BentoMLContainer.model_store`):
            BentoML modelstore, provided by DI Container.

    Returns:
        Runner instances for `bentoml.onnx` model

    Examples::
    """  # noqa
    return _ONNXRunner(
        tag=tag,
        backend=backend,
        gpu_device_id=gpu_device_id,
        disable_copy_in_default_stream=disable_copy_in_default_stream,
        providers=providers,
        session_options=session_options,
        resource_quota=resource_quota,
        batch_options=batch_options,
        model_store=model_store,
    )
