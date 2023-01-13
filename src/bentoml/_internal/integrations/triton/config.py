from __future__ import annotations

import os
import enum
import typing as t
from itertools import chain

import attr

from ...utils import bentoml_cattr
from ...utils import resolve_user_filepath

if t.TYPE_CHECKING:
    LogFormat = t.Literal["default", "ISO8601"]
    GrpcInferResponseCompressionLevel = t.Literal["none", "low", "medium", "high"]
    TraceLevel = t.Literal["OFF", "TIMESTAMPS", "TENSORS"]
else:
    LogFormat = str
    GrpcInferResponseCompressionLevel = str
    TraceLevel = str


def _to_list(inp: list[str] | str) -> list[str] | None:
    if inp is None:
        return
    if isinstance(inp, list):
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
    def from_string(cls, s: str | None) -> ModelControlMode | None:
        if s is None:
            return

        if not isinstance(s, str):
            raise ValueError(
                f"Can't convert given type {type(s)} to ModelControlMode, accepts strings only."
            )
        if s == "none":
            return ModelControlMode.NONE
        if s == "poll":
            return ModelControlMode.POLL
        if s == "explicit":
            return ModelControlMode.EXPLICIT
        raise ValueError(f"Invalid ModelControlMode: {s}")


@attr.frozen
class TritonServerConfig:
    """
    Triton Server configuration

    Note that BentoML will always run tritonserver with gRPC protocol, hence
    all config for HTTP will be ignored.

    Currently, Sagemaker and Vertex AI with Triton are yet to be implemented.
    """

    __omit_if_default__ = True

    _binary: str = attr.field(init=False, default="tritonserver")
    model_repository: t.List[str] = attr.field(converter=_to_list)

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

    # enable gRPC
    allow_grpc: bool = attr.field(
        init=False, default=None, converter=attr.converters.default_if_none(True)
    )
    # gRPC-related args
    grpc_port: int = attr.field(
        default=None, converter=attr.converters.default_if_none(8001)
    )
    reuse_grpc_port: bool = attr.field(default=None)
    # address to bind gRPC server to
    grpc_address: str = attr.field(default=None)
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

    # Allow user to enable Prometheus metrics from Triton
    allow_metrics: bool = attr.field(
        default=None, converter=attr.converters.default_if_none(True)
    )
    allow_gpu_metrics: bool = attr.field(default=None)
    allow_cpu_metrics: bool = attr.field(default=None)
    metrics_port: int = attr.field(
        default=None, converter=attr.converters.default_if_none(8002)
    )
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
    trace_level: t.List[TraceLevel] = attr.field(default=None, converter=_to_list)
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
        default=None, converter=ModelControlMode.from_string
    )
    repository_poll_secs: int = attr.field(default=None)
    #   Name of the model to be loaded on server startup. It may be
    # specified multiple times to add multiple models. To load ALL models
    # at startup, specify '*' as the model name with --load-model=* as the
    # ONLY --load-model argument, this does not imply any pattern
    # matching. Specifying --load-model=* in conjunction with another
    # --load-model argument will result in error. Note that this option will only
    # take effect if --model-control-mode=explicit is true.
    load_model: t.List[str] = attr.field(default=None, converter=_to_list)

    #   Path to backend shared library. Default is /opt/tritonserver/backends.
    backend_directory: str = attr.field(default=None)
    #   Path to repo agents. Default is /opt/tritonserver/repoagents.
    repoagent_directory: str = attr.field(default=None)

    def to_cli_args(self):
        cli: list[str] = [self._binary]
        margs = bentoml_cattr.unstructure(self)
        for arg, value in margs.items():
            opt = arg.replace("_", "-")
            if isinstance(value, list):
                cli.extend(
                    list(chain.from_iterable(map(lambda a: (f"--{opt}", a), value)))
                )
            else:
                cli.extend([f"--{opt}", str(value)])
        return cli
