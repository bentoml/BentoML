import shutil
import typing as t
import logging
from typing import TYPE_CHECKING

from simple_di import inject
from simple_di import Provide

import bentoml
from bentoml import Tag
from bentoml.exceptions import BentoMLException
from bentoml.exceptions import MissingDependencyException

from ..types import LazyType
from ..types import PathType
from ..models import SAVE_NAMESPACE
from ..utils.pkg import get_pkg_version
from ..runner.utils import Params
from ..runner.utils import get_gpu_memory
from .common.model_runner import BaseModelRunner
from ..configuration.containers import BentoMLContainer

SUPPORTED_ONNX_BACKEND: t.List[str] = ["onnxruntime", "onnxruntime-gpu"]
ONNX_EXT: str = ".onnx"

try:
    import onnx  # type: ignore
    import onnxruntime as ort  # type: ignore
except ImportError:
    raise MissingDependencyException(
        """\
`onnx` is required in order to use the module `bentoml.onnx`, do `pip install onnx`.
For more information, refers to https://onnx.ai/get-started.html
`onnxruntime` is also required by `bentoml.onnx`. Refer to https://onnxruntime.ai/ for
more information.
        """
    )

if TYPE_CHECKING:
    import torch
    import tensorflow as tf  # type: ignore

    from .. import external_typing as ext
    from ..models import ModelStore

    _ProviderType = t.List[t.Union[str, t.Tuple[str, t.Dict[str, t.Any]]]]
    _GPUProviderType = t.List[
        t.Tuple[
            t.Literal["CUDAExecutionProvider"],
            t.Union[t.Dict[str, t.Union[int, str, bool]], str],
        ]
    ]


try:
    import importlib.metadata as importlib_metadata
except ImportError:
    import importlib_metadata

_PACKAGE = ["onnxruntime", "onnxruntime-gpu"]
for p in _PACKAGE:
    try:
        _onnxruntime_version = importlib_metadata.version(p)
        break
    except importlib_metadata.PackageNotFoundError:
        pass

MODULE_NAME = "bentoml.onnx"

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


@inject
def load(
    tag: t.Union[str, Tag],
    backend: t.Optional[str] = "onnxruntime",
    providers: t.Optional[t.Union["_ProviderType", "_GPUProviderType"]] = None,
    session_options: t.Optional["ort.SessionOptions"] = None,  # type: ignore
    model_store: "ModelStore" = Provide[BentoMLContainer.model_store],
) -> "ort.InferenceSession":
    """
    Load a model from BentoML local modelstore with given name.

    Args:
        tag (:code:`Union[str, Tag]`):
            Tag of a saved model in BentoML local modelstore.
        backend (:code:`str`, `optional`, default to :code:`onnxruntime`):
            Different backend runtime supported by ONNX. Currently only accepted :obj:`onnxruntime`
            and :obj:`onnxruntime-gpu`.
        providers (`List[Union[str, Tuple[str, Dict[str, Any]]`, `optional`, default to :code:`None`):
            Different providers provided by users. By default BentoML will use :func:`onnxruntime.get_available_providers`
            when loading a model.
        session_options (`onnxruntime.SessionOptions`, `optional`, default to :code:`None`):
            SessionOptions per use case. If not specified, then default to :code:`None`.
        model_store (:mod:`~bentoml._internal.models.store.ModelStore`, default to :mod:`BentoMLContainer.model_store`):
            BentoML modelstore, provided by DI Container.

    Returns:
        :obj:`onnxruntime.InferenceSession`: an instance of ONNX model from BentoML modelstore.

    Examples:

    .. code-block:: python

        import bentoml

        model = bentoml.onnx.load(tag)

    """  # noqa
    model = model_store.get(tag)
    if model.info.module not in (MODULE_NAME, __name__):
        raise BentoMLException(
            f"Model {tag} was saved with module {model.info.module}, failed loading with {MODULE_NAME}."
        )
    model_file = model.path_of(f"{SAVE_NAMESPACE}{ONNX_EXT}")

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
        model_file,
        sess_options=session_options,
        providers=providers,
    )


def save(
    name: str,
    model: t.Union[onnx.ModelProto, PathType],
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
        model (Union[onnx.ModelProto, path-like object]):
            Instance of model to be saved.
        labels (:code:`Dict[str, str]`, `optional`, default to :code:`None`):
            user-defined labels for managing models, e.g. team=nlp, stage=dev
        custom_objects (:code:`Dict[str, Any]]`, `optional`, default to :code:`None`):
            user-defined additional python objects to be saved alongside the model,
            e.g. a tokenizer instance, preprocessor function, model configuration json
        metadata (:code:`Dict[str, Any]`, `optional`,  default to :code:`None`):
            Custom metadata for given model.
        model_store (:mod:`~bentoml._internal.models.store.ModelStore`, default to :mod:`BentoMLContainer.model_store`):
            BentoML modelstore, provided by DI Container.

    Returns:
        :obj:`~bentoml.Tag`: A :obj:`tag` with a format `name:version` where `name` is the user-defined model's name, and a generated `version` by BentoML.

    Examples:

    .. code-block:: python

        import bentoml

        import torch
        import torch.nn as nn

        class ExtendedModel(nn.Module):
            def __init__(self, D_in, H, D_out):
                # In the constructor we instantiate two nn.Linear modules and assign them as
                #  member variables.
                super(ExtendedModel, self).__init__()
                self.linear1 = nn.Linear(D_in, H)
                self.linear2 = nn.Linear(H, D_out)

            def forward(self, x, bias):
                # In the forward function we accept a Tensor of input data and an optional bias
                h_relu = self.linear1(x).clamp(min=0)
                y_pred = self.linear2(h_relu)
                return y_pred + bias


        N, D_in, H, D_out = 64, 1000, 100, 1
        x = torch.randn(N, D_in)
        model = ExtendedModel(D_in, H, D_out)

        input_names = ["x", "bias"]
        output_names = ["output1"]

        tmpdir = "/tmp/model"
        model_path = os.path.join(tmpdir, "test_torch.onnx")
        torch.onnx.export(
            model,
            (x, torch.Tensor([1.0])),
            model_path,
            input_names=input_names,
            output_names=output_names,
        )

        tag = bentoml.onnx.save("onnx_model", model_path, model_store=modelstore)
    """
    context: t.Dict[str, t.Any] = {
        "framework_name": "onnx",
        "pip_dependencies": [
            f"onnx=={get_pkg_version('onnx')}",
            f"onnxruntime=={_onnxruntime_version}",
        ],
    }

    with bentoml.models.create(
        name,
        module=MODULE_NAME,
        labels=labels,
        custom_objects=custom_objects,
        metadata=metadata,
        context=context,
    ) as _model:

        if isinstance(model, onnx.ModelProto):
            onnx.save_model(model, _model.path_of(f"{SAVE_NAMESPACE}{ONNX_EXT}"))
        else:
            shutil.copyfile(model, _model.path_of(f"{SAVE_NAMESPACE}{ONNX_EXT}"))

        return _model.tag


class _ONNXRunner(BaseModelRunner):
    def __init__(
        self,
        tag: t.Union[str, Tag],
        backend: str,
        gpu_device_id: int,
        disable_copy_in_default_stream: bool,
        providers: t.Optional["_ProviderType"],
        session_options: t.Optional["ort.SessionOptions"],  # type: ignore
        name: t.Optional[str] = None,
    ):
        super().__init__(tag, name=name)
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
                gpu_device_id,
                disable_copy_in_default_stream,
            )
        self._providers = providers
        self._session_options = session_options

    @staticmethod
    def _get_default_providers(
        gpu_device_id: int,
        disable_copy_in_default_stream: bool,
    ) -> "_ProviderType":
        if gpu_device_id != -1:
            _, free = get_gpu_memory(gpu_device_id)
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
        self,
        session_options: t.Optional["ort.SessionOptions"],
    ) -> "ort.SessionOptions":
        if session_options is not None:
            return session_options
        _session_options = ort.SessionOptions()
        _session_options.execution_mode = ort.ExecutionMode.ORT_PARALLEL
        _session_options.intra_op_num_threads = self._num_threads
        _session_options.inter_op_num_threads = self._num_threads
        _session_options.graph_optimization_level = (
            ort.GraphOptimizationLevel.ORT_ENABLE_EXTENDED
        )
        return _session_options

    @property
    def _num_threads(self) -> int:
        if self.resource_quota.on_gpu:
            return 1
        return max(round(self.resource_quota.cpu), 1)

    @property
    def num_replica(self) -> int:
        if self.resource_quota.on_gpu:
            return len(self.resource_quota.gpus)
        return 1

    def _setup(self) -> None:
        session_options = self._get_default_session_options(self._session_options)
        self._model = load(
            self._tag,
            backend=self._backend,
            providers=self._providers,
            session_options=session_options,
            model_store=self.model_store,
        )
        self._infer_func = getattr(self._model, "run")

    def _run_batch(  # type: ignore
        self,
        *args: t.Union[
            "ext.NpNDArray",
            "ext.PdDataFrame",
            "torch.Tensor",
            "tf.Tensor",
        ],
    ) -> t.Any:
        params = Params[
            t.Union[
                "ext.NpNDArray",
                "ext.PdDataFrame",
                "torch.Tensor",
                "tf.Tensor",
            ]
        ](*args)

        def _mapping(
            item: t.Union[
                "ext.NpNDArray",
                "ext.PdDataFrame",
                "torch.Tensor",
                "tf.Tensor",
            ]
        ) -> t.Any:
            if LazyType["ext.NpNDArray"]("numpy.ndarray").isinstance(item):
                pass
            elif LazyType["ext.PdDataFrame"]("pandas.DataFrame").isinstance(item):
                item = item.to_numpy()
            elif LazyType["tf.Tensor"]("tensorflow.Tensor").isinstance(item):
                item = item.numpy()  # type: ignore
            elif LazyType["torch.Tensor"]("torch.Tensor").isinstance(item):
                item = item.numpy()
            else:
                raise TypeError(
                    f"`_run_batch` of {self.__class__.__name__} only takes "
                    "`numpy.ndarray` or `pd.DataFrame` as input parameters"
                )
            return item

        params = params.map(_mapping)

        input_names = {
            i.name: val for i, val in zip(self._model.get_inputs(), params.args)
        }
        output_names = [_.name for _ in self._model.get_outputs()]
        return self._infer_func(output_names, input_names)


@inject
def load_runner(
    tag: t.Union[str, Tag],
    *,
    backend: str = "onnxruntime",
    gpu_device_id: int = -1,
    disable_copy_in_default_stream: bool = False,
    providers: t.Optional["_ProviderType"] = None,
    session_options: t.Optional["ort.SessionOptions"] = None,
    name: t.Optional[str] = None,
) -> "_ONNXRunner":
    """
    Runner represents a unit of serving logic that can be scaled horizontally to
    maximize throughput. `bentoml.onnx.load_runner` implements a Runner class that
    wrap around an ONNX model, which optimize it for the BentoML runtime.

    Args:
        tag (:code:`Union[str, Tag]`):
            Tag of a saved model in BentoML local modelstore.
        gpu_device_id (`int`, `optional`, default to :code:`-1`):
            GPU device ID. Currently only support CUDA.
        disable_copy_in_default_stream (`bool`, `optional`, default to :code:`False`):
            Whether to do copies in the default stream or use separate streams. Refers to `Execution Providers <https://onnxruntime.ai/docs/execution-providers/CUDA-ExecutionProvider.html#do_copy_in_default_stream>`_
            for more information.
        backend (:code:`str`, `optional`, default to :code:`onnxruntime`):
            Different backend runtime supported by ONNX. Currently only accepted :obj:`onnxruntime`
            and :obj:`onnxruntime-gpu`.
        providers (`List[Union[str, Tuple[str, Dict[str, Any]]`, `optional`, default to :code:`None`):
            Different providers provided by users. By default BentoML will use :obj:`CPUExecutionProvider` when
            loading a model.
        session_options (`onnxruntime.SessionOptions`, `optional`, default to :code:`None`):
            :obj:`SessionOptions` per use case. If not specified, then default to :code:`None`.

    Returns:
        :obj:`~bentoml._internal.runner.Runner`: Runner instances for :mod:`bentoml.onnx` model

    Examples:

    .. code-block:: python

        runner = bentoml.onnx.load_runner(
            tag, model_store=modelstore, backend="onnxruntime-gpu", gpu_device_id=0
        )
        runner.run_batch(data)
    """
    return _ONNXRunner(
        tag=tag,
        backend=backend,
        gpu_device_id=gpu_device_id,
        disable_copy_in_default_stream=disable_copy_in_default_stream,
        providers=providers,
        session_options=session_options,
        name=name,
    )
