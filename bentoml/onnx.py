import os
import shutil
import typing as t

from simple_di import Provide, inject

from ._internal.configuration.containers import BentoMLContainer
from ._internal.models import SAVE_NAMESPACE
from ._internal.runner import Runner
from ._internal.runner.utils import Params
from ._internal.utils import LazyLoader
from .exceptions import BentoMLException, MissingDependencyException

SUPPORTED_ONNX_BACKEND: t.List[str] = ["onnxruntime", "onnxruntime-gpu"]
ONNX_EXT: str = ".onnx"

if t.TYPE_CHECKING:  # pragma: no cover
    # pylint: disable=unused-import
    import numpy as np
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
`onnxruntime` is also required by `bentoml.onnx`. Refers to https://onnxruntime.ai/ for more information.
        """
    )

pd = LazyLoader("pd", globals(), "pandas")


# helper methods
def _yield_providers(iterable: t.Sequence[t.Any]) -> t.Generator[str, None, None]:
    if isinstance(iterable, tuple):
        yield iterable[0]
    elif isinstance(iterable, str):
        yield iterable
    else:
        yield from iterable


def flatten_list(lst: t.List[t.Any]) -> t.List[str]:
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
    providers: t.Optional[
        t.List[t.Union[str, t.Tuple[str, t.Dict[str, t.Any]]]]
    ] = None,
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
            Different providers provided by users. By default BentoML will use `CPUExecutionProvider` when
             loading a model
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
    model: t.Any,
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
        providers: t.Optional[t.List[t.Union[str, t.Tuple[str, t.Dict[str, t.Any]]]]],
        session_options: t.Optional["ort.SessionOptions"],
        resource_quota: t.Optional[t.Dict[str, t.Any]],
        batch_options: t.Optional[t.Dict[str, t.Any]],
        model_store: "ModelStore",
    ):
        super().__init__(tag, resource_quota, batch_options)
        model_info, model_file = _get_model_info(tag, model_store)
        self._model_info = model_info
        self._model_file = model_file
        self._model_store = model_store

        if backend not in SUPPORTED_ONNX_BACKEND:
            raise BentoMLException(
                f"'{backend}' runtime is currently not supported for ONNXModel"
            )
        if providers:
            if not all(i in ort.get_all_providers() for i in flatten_list(providers)):
                raise BentoMLException(
                    f"'{providers}' cannot be parsed by `onnxruntime`"
                )
        else:
            providers = ort.get_available_providers()

        self._backend = backend
        self._providers = providers

        self._session_options = (
            self._get_default_session_options()
            if not session_options
            else session_options
        )

    def _get_default_session_options(self) -> "ort.SessionOptions":
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
        if self.resource_quota.on_gpu:
            return 1
        return int(round(self.resource_quota.cpu))

    @property
    def num_replica(self) -> int:
        if self.resource_quota.on_gpu:
            return len(self.resource_quota.gpus)
        return 1

    # pylint: disable=arguments-differ,attribute-defined-outside-init
    def _setup(self) -> None:
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
        input_name = self._model.get_inputs()[0].name
        output_name = self._model.get_outputs()[0].name
        return self._infer_func([output_name], {input_name: params.args})[0]


@inject
def load_runner(
    tag: str,
    *,
    backend: str = "onnxruntime",
    providers: t.Optional[
        t.List[t.Union[str, t.Tuple[str, t.Dict[str, t.Any]]]]
    ] = None,
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
    if not providers:
        providers = ["CPUExecutionProvider"]
    return _ONNXRunner(
        tag=tag,
        backend=backend,
        providers=providers,
        session_options=session_options,
        resource_quota=resource_quota,
        batch_options=batch_options,
        model_store=model_store,
    )
