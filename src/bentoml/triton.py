from __future__ import annotations

import typing as t
import logging

import attr
from simple_di import inject as _inject
from simple_di import Provide as _Provide

from ._internal.utils import LazyLoader as _LazyLoader
from ._internal.utils import cached_property as _cached_property
from ._internal.configuration import get_debug_mode as _get_debug_mode
from ._internal.runner.runner import RunnerMethod as _RunnerMethod
from ._internal.runner.runner import AbstractRunner as _AbstractRunner
from ._internal.runner.runner import object_setattr as _object_setattr
from ._internal.runner.runnable import RunnableMethodConfig as _RunnableMethodConfig
from ._internal.runner.runner_handle import DummyRunnerHandle as _DummyRunnerHandle
from ._internal.configuration.containers import BentoMLContainer as _BentoMLContainer
from ._internal.runner.runner_handle.remote import TRITON_EXC_MSG as _TRITON_EXC_MSG
from ._internal.runner.runner_handle.remote import (
    handle_triton_exception as _handle_triton_exception,
)

if t.TYPE_CHECKING:
    import tritonclient.grpc.aio as _tritongrpcclient
    import tritonclient.http.aio as _tritonhttpclient

    from ._internal.runner.runner_handle import RunnerHandle

    _P = t.ParamSpec("_P")

    _LogFormat = t.Literal["default", "ISO8601"]
    _GrpcInferResponseCompressionLevel = t.Literal["none", "low", "medium", "high"]
    _TraceLevel = t.Literal["OFF", "TIMESTAMPS", "TENSORS"]
    _RateLimit = t.Literal["execution_count", "off"]
    _TritonServerType = t.Literal["grpc", "http"]

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
        "_tritongrpcclient", globals(), "tritonclient.grpc.aio", exc_msg=_TRITON_EXC_MSG
    )
    _tritonhttpclient = _LazyLoader(
        "_tritonhttpclient", globals(), "tritonclient.http.aio", exc_msg=_TRITON_EXC_MSG
    )

_logger = logging.getLogger(__name__)

__all__ = ["Runner"]


@attr.define(slots=False, frozen=True, eq=False)
class _TritonRunner(_AbstractRunner):
    repository_path: str

    tritonserver_type: _TritonServerType = attr.field(
        default="grpc", validator=attr.validators.in_(["grpc", "http"])
    )
    cli_args: list[str] = attr.field(factory=list)

    _runner_handle: RunnerHandle = attr.field(init=False, factory=_DummyRunnerHandle)

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

    def __init__(
        self,
        name: str,
        model_repository: str,
        tritonserver_type: _TritonServerType = "grpc",
        cli_args: list[str] | None = None,
    ):
        if cli_args is None:
            cli_args = []

        cli_args.append(f"--model-repository={model_repository}")

        if tritonserver_type == "http":
            cli_args.extend(
                [
                    "--allow-grpc=False",
                    "--http-address=127.0.0.1",
                ]
            )
        elif tritonserver_type == "grpc":
            cli_args.extend(
                [
                    "--reuse-grpc-port=1",
                    "--allow-http=False",
                    "--grpc-address=0.0.0.0",
                ]
            )

        # default settings, disable metrics
        cli_args.extend([f"--log-verbose={1 if _get_debug_mode() else 0}"])

        if not all(s.startswith("--") for s in cli_args):
            raise ValueError(
                "cli_args should be a list of strings starting with '--' for TritonRunner."
            )

        self.__attrs_init__(
            name=name,
            models=None,
            resource_config=None,
            runnable_class=self.__class__,
            repository_path=model_repository,
            tritonserver_type=tritonserver_type,
            cli_args=cli_args,
            embedded=False,  # NOTE: TritonRunner shouldn't be used as embedded.
        )

    @_cached_property
    def protocol_address(self):
        from ._internal.utils import reserve_free_port

        if self.tritonserver_type == "http":
            with reserve_free_port(host="127.0.0.1") as port:
                pass
            return f"127.0.0.1:{port}"
        elif self.tritonserver_type == "grpc":
            with reserve_free_port(host="0.0.0.0", enable_so_reuseport=True) as port:
                pass
            return f"0.0.0.0:{port}"
        else:
            raise ValueError(f"Invalid Triton Server type: {self.tritonserver_type}")

    def init_local(self, quiet: bool = False) -> None:
        _logger.warning(
            "TritonRunner '%s' will not be available for development mode.", self.name
        )

    def init_client(
        self,
        handle_class: type[RunnerHandle] | None = None,
        *args: t.Any,
        **kwargs: t.Any,
    ):
        from ._internal.runner.runner_handle.remote import TritonRunnerHandle

        if handle_class is None:
            handle_class = TritonRunnerHandle

        super().init_client(handle_class=handle_class, *args, **kwargs)

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
    ) -> _RunnerMethod[
        t.Any, _P, _tritongrpcclient.InferResult | _tritonhttpclient.InferResult
    ]:
        ...

    def __getattr__(self, item: str) -> t.Any:
        from ._internal.runner.runner_handle.remote import TritonRunnerHandle

        if isinstance(self._runner_handle, TritonRunnerHandle):
            if item in self._runner_handle.client_methods:
                # NOTE: auto wrap triton methods to its respective clients
                if self.tritonserver_type == "grpc":
                    return _handle_triton_exception(
                        getattr(self._runner_handle.grpc_client, item)
                    )
                else:
                    return _handle_triton_exception(
                        getattr(self._runner_handle.http_client, item)
                    )
            else:
                # if given item is not a client method, then we assume it is a model name.
                # Hence, we will return a RunnerMethod that will be responsible for this model handle.
                RT = (
                    _tritonhttpclient.InferResult
                    if self.tritonserver_type == "http"
                    else _tritongrpcclient.InferResult
                )
                return _RunnerMethod[t.Any, _P, RT](
                    runner=self,
                    name=item,
                    config=_RunnableMethodConfig(batchable=True, batch_dim=(0, 0)),
                    max_batch_size=0,
                    max_latency_ms=10000,
                )

        return super().__getattribute__(item)


Runner = _TritonRunner
