from __future__ import annotations

import enum
import shutil
import typing as t
import logging
import itertools

import attr
from simple_di import inject as _inject
from simple_di import Provide as _Provide

from .exceptions import StateException as _StateException
from ._internal.types import LazyType as _LazyType
from ._internal.utils import LazyLoader as _LazyLoader
from ._internal.configuration import get_debug_mode as _get_debug_mode
from ._internal.runner.runner import RunnerMethod as _RunnerMethod
from ._internal.runner.runner import AbstractRunner as _AbstractRunner
from ._internal.runner.runner import object_setattr as _object_setattr
from ._internal.runner.runnable import RunnableMethodConfig as _RunnableMethodConfig
from ._internal.runner.runner_handle import DummyRunnerHandle as _DummyRunnerHandle
from ._internal.configuration.containers import BentoMLContainer as _BentoMLContainer
from ._internal.runner.runner_handle.remote import (
    handle_triton_exception as _handle_triton_exception,
)

if t.TYPE_CHECKING:
    import tritonclient.grpc.aio as _tritongrpcclient

    from ._internal.runner.runner_handle import RunnerHandle

    _P = t.ParamSpec("_P")

    _LogFormat = t.Literal["default", "ISO8601"]
    _GrpcInferResponseCompressionLevel = t.Literal["none", "low", "medium", "high"]
    _TraceLevel = t.Literal["OFF", "TIMESTAMPS", "TENSORS"]
    _RateLimit = t.Literal["execution_count", "off"]

    _ClientMethod = t.Literal[
        "get_cuda_shared_memory_status",
        "get_inference_statistics",
        "get_log_settings",
        "get_model_config",
        "get_model_metadata",
        "get_model_repository_index",
        "get_server_metadata",
        "get_system_shared_memory_status",
        "get_trace_settings",
        "infer",
        "is_model_ready",
        "is_server_live",
        "is_server_ready",
        "load_model",
        "register_cuda_shared_memory",
        "register_system_shared_memory",
        "stream_infer",
        "unload_model",
        "unregister_cuda_shared_memory",
        "unregister_system_shared_memory",
        "update_log_settings",
        "update_trace_settings",
    ]
    _ModelName = t.Annotated[str, t.LiteralString]

else:
    _P = t.TypeVar("_P")

    _LogFormat = _GrpcInferResponseCompressionLevel = _TraceLevel = _RateLimit = str

    _tritongrpcclient = _LazyLoader(
        "_tritongrpcclient",
        globals(),
        "tritonclient.grpc.aio",
        exc_msg="tritonclient is required to use triton with BentoML. Install with 'pip install \"tritonclient[grpc]>=2.29.0\"'.",
    )

_logger = logging.getLogger(__name__)

__all__ = ["Runner"]


@attr.define(slots=False, frozen=True, eq=False)
class _TritonRunner(_AbstractRunner):
    repository_path: str

    _runner_handle: RunnerHandle = attr.field(init=False, factory=_DummyRunnerHandle)

    def _init(self, handle_class: t.Type[RunnerHandle]) -> None:
        if not isinstance(self._runner_handle, _DummyRunnerHandle):
            raise _StateException("Runner already initialized")

        _object_setattr(self, "_runner_handle", handle_class(self))

    @_inject
    async def runner_handle_is_ready(
        self,
        timeout: int = _Provide[
            _BentoMLContainer.api_server_config.runner_probe.timeout
        ],
    ) -> bool:
        """
        Check if given runner handle is ready. This will be used as readiness probe in Kubernetes.
        """
        return await self._runner_handle.is_ready(timeout)

    def __init__(self, name: str, model_repository: str):
        self.__attrs_init__(
            name=name,
            models=None,
            resource_config=None,
            runnable_class=self.__class__,
            repository_path=model_repository,
        )

    def init_local(self, quiet: bool = False) -> None:
        _logger.warning(
            "TritonRunner '%s' will not be available for development mode.", self.name
        )

    def init_client(self):
        from ._internal.runner.runner_handle.remote import TritonRunnerHandle

        self._init(TritonRunnerHandle)

    def destroy(self):
        _object_setattr(self, "_runner_handle", _DummyRunnerHandle())

    # Even though the below overload overlaps, it is ok to ignore the warning since types
    # for TritonRunner can handle both function from client and LiteralString from model name.
    @t.overload
    def __getattr__(self, item: t.Literal["__attrs_init__"]) -> t.Callable[..., None]:  # type: ignore (overload warning)
        ...

    @t.overload
    def __getattr__(
        self, item: _ClientMethod
    ) -> t.Callable[..., t.Coroutine[t.Any, t.Any, t.Any]]:
        ...

    @t.overload
    def __getattr__(
        self, item: _ModelName
    ) -> _RunnerMethod[t.Any, _P, _tritongrpcclient.InferResult]:
        ...

    def __getattr__(self, item: str) -> t.Any:
        from ._internal.runner.runner_handle.remote import TritonRunnerHandle

        if isinstance(self._runner_handle, TritonRunnerHandle):
            if item in self._runner_handle.client_methods:
                return _handle_triton_exception(
                    getattr(self._runner_handle.client, item)
                )
            else:
                # if given item is not a client method, then we assume it is a model name.
                # Hence, we will return a RunnerMethod that will be responsible for this model handle.
                return _RunnerMethod[t.Any, _P, _tritongrpcclient.InferResult](
                    runner=self,
                    name=item,
                    config=_RunnableMethodConfig(batchable=False, batch_dim=(0, 0)),
                    max_batch_size=0,
                    max_latency_ms=10000,
                )

        return super().__getattribute__(item)


def _to_mut_iterable(
    inp: tuple[str, ...] | list[str] | str
) -> tuple[str, ...] | list[str] | None:
    if inp is None:
        return
    if isinstance(inp, (list, tuple)):
        return inp
    elif isinstance(inp, str):
        return [inp]
    else:
        raise ValueError(f"Unknown type: {type(inp)}")


Runner = _TritonRunner


class _ModelControlMode(enum.Enum):
    NONE = "none"
    POLL = "poll"
    EXPLICIT = "explicit"

    @classmethod
    def from_type(cls, s: t.Any) -> _ModelControlMode:
        if s is None:
            return _ModelControlMode.NONE
        elif isinstance(s, _ModelControlMode):
            return s

        if isinstance(s, (tuple, list)):
            # parsed from CLI or SDK
            s = s[0]

        if not isinstance(s, str):
            raise ValueError(
                f"Can't convert given type {type(s)} [value: {s}] to ModelControlMode, accepts strings only."
            )
        if s == "none":
            return _ModelControlMode.NONE
        if s == "poll":
            return _ModelControlMode.POLL
        if s == "explicit":
            return _ModelControlMode.EXPLICIT
        raise ValueError(
            f"Invalid ModelControlMode: {s}, accepted value: ['none', 'poll', 'explicit']"
        )


def _find_triton_binary():
    binary = shutil.which("tritonserver")
    if binary is None:
        raise RuntimeError(
            "'tritonserver' is not found on PATH. Make sure to include the compiled binary in PATH to proceed.\nIf you are running this inside a container, make sure to use the official Triton container image as a 'base_image'. See https://catalog.ngc.nvidia.com/orgs/nvidia/containers/tritonserver."
        )
    return binary


@attr.frozen
class TritonServerHandle:
    """
    Triton Server handle. This is a dataclass to handle CLI arguments that pass to 'tritonserver' binary.

    Note that BentoML will always run tritonserver with gRPC protocol, hence
    all config for HTTP will be ignored.

    Currently, Sagemaker and Vertex AI with Triton are yet to be implemented.
    """

    __omit_if_default__ = True
    _binary: str = attr.field(init=False, factory=_find_triton_binary)

    model_repository: t.List[str] = attr.field(converter=_to_mut_iterable)

    # arguments that is set by default by BentoML
    allow_http: bool = attr.field(
        init=False, default=None, converter=attr.converters.default_if_none(False)
    )
    allow_grpc: bool = attr.field(
        init=False, default=None, converter=attr.converters.default_if_none(True)
    )

    # address to bind gRPC server to
    grpc_address: str = attr.field(
        default=None, converter=attr.converters.default_if_none("0.0.0.0")
    )
    grpc_port: int = attr.field(default=None)
    # Always reuse gRPC port so that we can spawn multiple tritonserver gRPC server.
    reuse_grpc_port: int = attr.field(
        default=None, converter=attr.converters.default_if_none(1)
    )

    # TODO: support Sagemaker and Vertex AI
    # TODO: support rate limit and memory cache management

    # log-related args
    log_verbose: int = attr.field(
        default=None,
        converter=attr.converters.default_if_none(1 if _get_debug_mode() else 0),
    )
    log_info: bool = attr.field(default=None)
    log_warning: bool = attr.field(default=None)
    log_error: bool = attr.field(default=None)
    log_format: _LogFormat = attr.field(default=None)
    # if specified, then stream logs to a file, otherwise to stdout.
    log_file: str = attr.field(default=None)
    # identifier of this triton server instance
    id: str = attr.field(default=None)
    # whether to exit on error during initialization
    exit_on_error: bool = attr.field(default=None)

    # gRPC-related args
    #   The maximum number of inference request/response objects
    # that remain allocated for reuse. As long as the number of in-flight
    # requests doesn't exceed this value there will be no
    # allocation/deallocation of request/response objects.
    grpc_infer_allocation_pool_size: int = attr.field(default=None)
    # Whether to use SSL authentication for gRPC request. Default is False
    grpc_use_ssl: bool = attr.field(default=None)
    # Whether to use mututal SSL authentication. Default is False
    grpc_use_ssl_mutual: bool = attr.field(default=None)
    # PEM-encoded server, root certificate and key
    grpc_server_cert: str = attr.field(default=None)
    grpc_server_key: str = attr.field(default=None)
    grpc_root_cert: str = attr.field(default=None)
    # compression level to be used while returning inference results
    grpc_infer_response_compression_level: _GrpcInferResponseCompressionLevel = (
        attr.field(default=None)
    )
    #   Period in miliseconds after which a keepalive ping is sent on the transport
    # default is 7200000 (2 hours)
    grpc_keepalive_time: int = attr.field(default=None)
    #   Allows keepalive pings to be sent even if there are no calls
    # in flight (0 : false; 1 : true). Default is 0 (false).
    grpc_keepalive_permit_without_calls: int = attr.field(default=None)
    #   The maximum number of pings that can be sent when there is
    # no data/header frame to be sent. gRPC Core will not continue sending
    # pings if we run over the limit. Setting it to 0 allows sending pings
    # without such a restriction. Default is 2.
    grpc_http2_max_pings_without_data: int = attr.field(default=None)
    #   If there are no data/header frames being sent on the
    # transport, this channel argument on the server side controls the minimum
    # time (in milliseconds) that gRPC Core would expect between receiving
    # successive pings. If the time between successive pings is less than
    # this time, then the ping will be considered a bad ping from the peer.
    # Such a ping counts as a ‘ping strike’. Default is 300000 (5 minutes).
    grpc_http2_min_recv_ping_interval_without_data: int = attr.field(default=None)
    #   Maximum number of bad pings that the server will tolerate
    # before sending an HTTP2 GOAWAY frame and closing the transport.
    # Setting it to 0 allows the server to accept any number of bad pings.
    # Default is 2.
    grpc_http2_max_ping_strikes: int = attr.field(default=None)

    # trace-related args
    #   Set the file where trace output will be saved. If
    # --triton-options trace-log-frequency is also specified, this argument value will be the
    # prefix of the files to save the trace output.
    trace_file: str = attr.field(default=None)
    #   Specify a trace level. OFF to disable tracing, TIMESTAMPS to
    # trace timestamps, TENSORS to trace tensors. It may be specified
    # multiple times to trace multiple informations. Default is OFF.
    trace_level: t.List[_TraceLevel] = attr.field(
        default=None, converter=attr.converters.default_if_none(factory=list)
    )
    #   Set the trace sampling rate. Default is 1000
    trace_rate: int = attr.field(default=None)
    #   Number of traces to be sampled. If set to -1, # of traces are unlimted. Default
    # to -1
    trace_count: int = attr.field(default=None)
    #   Set the trace log frequency. If the value is 0, Triton will
    # only log the trace output to <trace-file> when shutting down.
    # Otherwise, Triton will log the trace output to <trace-file>.<idx> when it
    # collects the specified number of traces. For example, if the log
    # frequency is 100, when Triton collects the 100-th trace, it logs the
    # traces to file <trace-file>.0, and when it collects the 200-th trace,
    # it logs the 101-th to the 200-th traces to file <trace-file>.1.
    # Default is 0.
    trace_log_frequency: int = attr.field(default=None)

    # Model management args
    #  Specify the mode for model management. Options are "none",
    # "poll" and "explicit". The default is "none". For "none", the server
    # will load all models in the model repository(s) at startup and will
    # not make any changes to the load models after that. For "poll", the
    # server will poll the model repository(s) to detect changes and will
    # load/unload models based on those changes. The poll rate is
    # controlled by 'repository-poll-secs'. For "explicit", model load and unload
    # is initiated by using the model control APIs, and only models
    # specified with --triton-options load-model will be loaded at startup.
    model_control_mode: _ModelControlMode = attr.field(
        default="none", converter=_ModelControlMode.from_type
    )
    repository_poll_secs: int = attr.field(default=None)
    #   Name of the model to be loaded on server startup. It may be
    # specified multiple times to add multiple models. To load ALL models
    # at startup, specify '*' as the model name with --load-model=* as the
    # ONLY --triton-options load-model argument, this does not imply any pattern
    # matching. Specifying --triton-options load-model=* in conjunction with another
    # --triton-options load-model argument will result in error. Note that this option will only
    # take effect if --triton-options model-control-mode=explicit is true.
    load_model: t.List[str] = attr.field(default=None, converter=_to_mut_iterable)
    #   Specify the mode for rate limiting. Options are
    # "execution_count" and "off". The default is "off". For "execution_count", the
    # server will determine the instance using configured priority and the
    # number of time the instance has been used to run inference. The
    # inference will finally be executed once the required resources are
    # available. For "off", the server will ignore any rate limiter config and
    # run inference as soon as an instance is ready.
    rate_limit: _RateLimit = attr.field(default=None)
    # 	The number of resources available to the server. The format # of this flag is
    # --rate-limit-resource=<resource_name>:<count>:<device>. The <device> is optional and if not listed will be applied to
    # every device. If the resource is specified as "GLOBAL" in the model
    # configuration the resource is considered shared among all the devices
    # in the system. The <device> property is ignored for such resources.
    # This flag can be specified multiple times to specify each resources
    # and their availability. By default, the max across all instances
    # that list the resource is selected as its availability. The values for
    # this flag is case-insensitive.
    rate_limit_resource: t.List[str] = attr.field(
        default=None, converter=_to_mut_iterable
    )
    # The total byte size that can be allocated as pinned system
    # memory. If GPU support is enabled, the server will allocate pinned
    # system memory to accelerate data transfer between host and devices
    # until it exceeds the specified byte size. If 'numa-node' is configured
    # via --host-policy, the pinned system memory of the pool size will be
    # allocated on each numa node. This option will not affect the
    # allocation conducted by the backend frameworks. Default is 256 MB.
    pinned_memory_pool_byte_size: int = attr.field(default=None)
    # The total byte size that can be allocated as CUDA memory for
    # the GPU device. If GPU support is enabled, the server will allocate
    # CUDA memory to minimize data transfer between host and devices
    # until it exceeds the specified byte size. This option will not affect
    # the allocation conducted by the backend frameworks. The argument
    # should be 2 integers separated by colons in the format <GPU device
    # ID>:<pool byte size>. This option can be used multiple times, but only
    # once per GPU device. Subsequent uses will overwrite previous uses for
    # the same GPU device. Default is 64 MB.
    cuda_memory_pool_byte_size: t.List[str] = attr.field(
        default=None, converter=_to_mut_iterable
    )
    # The size in bytes to allocate for a request/response cache.
    # When non-zero, Triton allocates the requested size in CPU memory and
    # shares the cache across all inference requests and across all
    # models. For a given model to use request caching, the model must enable
    # request caching in the model configuration. By default, no model uses
    # request caching even if the request cache is enabled with the
    # --response-cache-byte-size flag. Default is 0.
    response_cache_byte_size: int = attr.field(default=None)
    # The minimum supported CUDA compute capability. GPUs that
    # don't support this compute capability will not be used by the server.
    min_supported_compute_capability: float = attr.field(default=None)
    # Timeout (in seconds) when exiting to wait for in-flight
    # inferences to finish. After the timeout expires the server exits even if
    # inferences are still in flight.
    exit_timeout_secs: int = attr.field(default=None)
    # The global directory searched for backend shared libraries.
    # Default is '/opt/tritonserver/backends'.
    backend_directory: str = attr.field(default=None)
    # The global directory searched for repository agent shared
    # libraries. Default is '/opt/tritonserver/repoagents'.
    repoagent_directory: str = attr.field(default=None)
    # The number of threads used to accelerate copies and other
    # operations required to manage input and output tensor contents.
    # Default is 0.
    buffer_manager_thread_count: int = attr.field(default=None)
    # The number of threads used to concurrently load models in
    # model repositories. Default is 2*<num_cpu_cores>.
    model_load_thread_count: int = attr.field(default=None)
    # Specify a backend-specific configuration setting. The format
    # of this flag is --backend-config=<backend_name>,<setting>=<value>.
    # Where <backend_name> is the name of the backend, such as 'tensorrt'.
    backend_config: t.List[str] = attr.field(default=None, converter=_to_mut_iterable)
    # Specify a host policy setting associated with a policy name.
    # The format of this flag is
    # --host-policy=<policy_name>,<setting>=<value>. Currently supported settings are 'numa-node', 'cpu-cores'.
    # Note that 'numa-node' setting will affect pinned memory pool behavior,
    # see --pinned-memory-pool for more detail.
    host_policy: t.List[str] = attr.field(default=None, converter=_to_mut_iterable)
    # Specify the limit on GPU memory usage as a fraction. If
    # model loading on the device is requested and the current memory usage
    # exceeds the limit, the load will be rejected. If not specified, the
    # limit will not be set.
    model_load_gpu_limit: t.List[str] = attr.field(
        default=None, converter=_to_mut_iterable
    )

    def to_cli_args(self):
        from ._internal.utils import bentoml_cattr

        resolved: dict[str, t.Any] = bentoml_cattr.unstructure(self)

        cli: list[str] = []
        for arg, value in resolved.items():
            if _LazyType["list[str]"](list).isinstance(value) or _LazyType[
                "tuple[str, ...]"
            ](tuple).isinstance(value):
                cli.extend(
                    list(
                        itertools.chain.from_iterable(
                            map(lambda a: (f"--{arg.replace('_', '-')}", a), value)
                        )
                    )
                )
            else:
                cli.extend([f"--{arg.replace('_', '-')}", str(value)])
        return cli

    def with_args(self, **kwargs: t.Any):
        return attr.evolve(self, **kwargs)

    def to_dict(self, omit_if_default: bool = False):
        if not omit_if_default:
            # we want to use the default cattr.Converter to return all default values.
            import cattr
        else:
            # by default, this class is set to omit all default values
            # set from Python to use default values that are set by tritonserver.
            from ._internal.utils import bentoml_cattr as cattr

        return cattr.unstructure(self)

    @property
    def executable(self):
        return self._binary

    @property
    def args(self):
        return self.to_cli_args()
