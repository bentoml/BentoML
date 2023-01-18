from __future__ import annotations

import os
import enum
import shutil
import typing as t
import logging
import urllib.parse
from itertools import chain

import fs
import attr
import fs.errors
import fs.opener
import fs.opener.errors

from ._internal.utils import bentoml_cattr
from ._internal.utils import resolve_user_filepath
from ._internal.runner.runner import RunnerMeta
from ._internal.runner.runner import RunnerMethod
from ._internal.runner.runner import object_setattr
from ._internal.runner.runnable import RunnableMethodConfig
from ._internal.runner.runner_handle.remote import TritonRunnerHandle

if t.TYPE_CHECKING:
    from fs.base import FS
    from fs.opener.registry import Registry

    P = t.ParamSpec("P")
    R = t.TypeVar("R")

    LogFormat = t.Literal["default", "ISO8601"]
    GrpcInferResponseCompressionLevel = t.Literal["none", "low", "medium", "high"]
    TraceLevel = t.Literal["OFF", "TIMESTAMPS", "TENSORS"]
else:
    LogFormat = str
    GrpcInferResponseCompressionLevel = str
    TraceLevel = str

logger = logging.getLogger(__name__)

__all__ = ["Runner"]


@attr.define(slots=False, frozen=True, eq=False)
class TritonRunner(RunnerMeta):
    _fs: FS
    repository_path: str

    def __init__(
        self,
        name: str,
        model_repository: str,
        *,
        fs_protocol: str | None = None,
    ):
        # TODO: Support configuration (P1)

        # The logic below mimic the behaviour of Exportable.import_from
        # To determine models inside given model repository.
        try:
            parsed_model_repository = fs.opener.parse(model_repository)
        except fs.opener.errors.ParseError:
            if fs_protocol is None:
                fs_protocol = "osfs"
                resource: str = (
                    model_repository
                    if os.sep == "/"
                    else model_repository.replace(os.sep, "/")
                )
            else:
                resource: str = ""
        else:
            if fs_protocol is not None:
                raise ValueError(
                    "An FS URL was passed as the output path; all additional information should be passed as part of the URL."
                )

            fs_protocol = parsed_model_repository.protocol
            resource: str = parsed_model_repository.resource

        if fs_protocol not in t.cast("Registry", fs.opener.registry).protocols:
            if fs_protocol == "s3":
                raise ValueError(
                    "Tried to open an S3 url but the protocol is not registered; did you 'pip install \"bentoml[aws]\"'?"
                )
            else:
                raise ValueError(
                    f"Unknown or unsupported protocol {fs_protocol}. Some supported protocols are 'ftp', 's3', and 'osfs'."
                )

        is_os_path = fs_protocol in [
            "temp",
            "osfs",
            "userdata",
            "userconf",
            "sitedata",
            "siteconf",
            "usercache",
            "userlog",
        ]

        if is_os_path:
            try:
                repo_fs = fs.open_fs(f"{fs_protocol}://{resource}")
            except fs.errors.CreateFailed:
                raise ValueError(f"Path {model_repository} does not exist.")
        else:
            resource = urllib.parse.quote(resource)
            repo_fs = fs.open_fs(f"{fs_protocol}://{resource}")

        self.__attrs_init__(
            name=name, models=[], repository_path=model_repository, fs=repo_fs
        )

        # List of models inside given model repository.
        for model in repo_fs.listdir("/"):
            object_setattr(
                self,
                model,
                RunnerMethod(
                    runner=self,
                    name=model,
                    # TODO: support config for Triton runners.
                    config=RunnableMethodConfig(batchable=False, batch_dim=(0, 0)),
                    max_batch_size=0,
                    max_latency_ms=100000,
                ),
            )

    def init_local(self, quiet: bool = False) -> None:
        raise ValueError(
            "'init_local' is not supported for TritonRunner as this is intended to be used with Triton Inference Server."
        )

    def init_client(self):
        self._init(TritonRunnerHandle)


def _to_mut_iterable(
    inp: tuple[str, ...] | list[str] | str
) -> tuple[str, ...] | list[str] | None:
    if inp is None:
        return
    if isinstance(inp, (list, tuple)):
        return inp
    elif isinstance(inp, str):
        if os.path.exists(inp):
            inp = resolve_user_filepath(inp, ctx=None)
        return [inp]
    else:
        raise ValueError(f"Unknown type: {type(inp)}")


class ModelControlMode(enum.Enum):
    NONE = "none"
    POLL = "poll"
    EXPLICIT = "explicit"

    @classmethod
    def from_type(cls, s: t.Any) -> ModelControlMode:
        if s is None:
            return ModelControlMode.NONE

        if isinstance(s, (tuple, list)):
            # parsed from CLI or SDK
            s = s[0]

        if not isinstance(s, str):
            raise ValueError(
                f"Can't convert given type {type(s)} [value: {s}] to ModelControlMode, accepts strings only."
            )
        if s == "none":
            return ModelControlMode.NONE
        if s == "poll":
            return ModelControlMode.POLL
        if s == "explicit":
            return ModelControlMode.EXPLICIT
        raise ValueError(f"Invalid ModelControlMode: {s}")


Runner = TritonRunner


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
    grpc_port: int = attr.field(
        default=None, converter=attr.converters.default_if_none(8001)
    )
    # Always reuse gRPC port so that we can spawn multiple tritonserver gRPC server.
    reuse_grpc_port: int = attr.field(
        default=None, converter=attr.converters.default_if_none(1)
    )
    # by default, expose the Triton metrics server
    allow_metrics: bool = attr.field(
        default=None, converter=attr.converters.default_if_none(True)
    )
    metrics_port: int = attr.field(
        default=None, converter=attr.converters.default_if_none(8002)
    )

    # TODO: support Sagemaker and Vertex AI
    # TODO: support rate limit and memory cache management

    # log-related args
    log_verbose: int = attr.field(default=None)
    log_info: bool = attr.field(default=None)
    log_warning: bool = attr.field(default=None)
    log_error: bool = attr.field(default=None)
    log_format: LogFormat = attr.field(default=None)
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
    grpc_infer_response_compression_level: GrpcInferResponseCompressionLevel = (
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

    # Metrics server related (Prometheus)
    allow_gpu_metrics: bool = attr.field(default=None)
    allow_cpu_metrics: bool = attr.field(default=None)
    #   Metrics will be collected once every <metrics-interval-ms> milliseconds.
    # Default is 2000 milliseconds.
    metrics_interval_ms: float = attr.field(default=None)

    # trace-related args
    #   Set the file where trace output will be saved. If
    # --trace-log-frequency is also specified, this argument value will be the
    # prefix of the files to save the trace output.
    trace_file: str = attr.field(default=None)
    #   Specify a trace level. OFF to disable tracing, TIMESTAMPS to
    # trace timestamps, TENSORS to trace tensors. It may be specified
    # multiple times to trace multiple informations. Default is OFF.
    trace_level: t.List[TraceLevel] = attr.field(
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
    # specified with --load-model will be loaded at startup.
    model_control_mode: ModelControlMode = attr.field(
        default=None, converter=ModelControlMode.from_type
    )
    repository_poll_secs: int = attr.field(default=None)
    #   Name of the model to be loaded on server startup. It may be
    # specified multiple times to add multiple models. To load ALL models
    # at startup, specify '*' as the model name with --load-model=* as the
    # ONLY --load-model argument, this does not imply any pattern
    # matching. Specifying --load-model=* in conjunction with another
    # --load-model argument will result in error. Note that this option will only
    # take effect if --model-control-mode=explicit is true.
    load_model: t.List[str] = attr.field(
        default=None, converter=attr.converters.default_if_none(factory=list)
    )

    #   Path to backend shared library. Default is /opt/tritonserver/backends.
    backend_directory: str = attr.field(default=None)
    #   Path to repo agents. Default is /opt/tritonserver/repoagents.
    repoagent_directory: str = attr.field(default=None)

    def to_cli_args(self):
        cli: list[str] = []
        margs = bentoml_cattr.unstructure(self)
        for arg, value in margs.items():
            opt = arg.replace("_", "-")
            if isinstance(value, (list, tuple)):
                cli.extend(
                    list(chain.from_iterable(map(lambda a: (f"--{opt}", a), value)))
                )
            else:
                cli.extend([f"--{opt}", str(value)])
        logger.debug("tritonserver cmd: '%s %s'", self.executable, " ".join(cli))
        return cli

    def to_dict(self, omit_if_default: bool = False):
        # by default, this class is set to omit all default values
        # set from Python to use default values that are set by tritonserver.
        converter_mod = bentoml_cattr

        if not omit_if_default:
            # we want to use the default cattr.Converter
            # to return all default values.
            import cattr

            converter_mod = cattr

        return converter_mod.unstructure(self)

    @property
    def executable(self):
        return self._binary

    @property
    def args(self):
        return self.to_cli_args()
