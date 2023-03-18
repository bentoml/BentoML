from __future__ import annotations

import os
import json
import time
import pickle
import typing as t
import asyncio
import logging
import functools
import traceback
from json.decoder import JSONDecodeError
from urllib.parse import urlparse

from . import RunnerHandle
from ..utils import Params
from ..utils import PAYLOAD_META_HEADER
from ...utils import LazyLoader
from ...context import component_context
from ..container import Payload
from ...utils.uri import uri_to_path
from ....exceptions import RemoteException
from ....exceptions import ServiceUnavailable
from ...configuration.containers import BentoMLContainer

TRITON_EXC_MSG = "tritonclient is required to use triton with BentoML. Install with 'pip install \"tritonclient[all]>=2.29.0\"'."

if t.TYPE_CHECKING:
    import yarl
    import tritonclient.grpc.aio as tritongrpcclient
    import tritonclient.http.aio as tritonhttpclient
    from aiohttp import BaseConnector
    from aiohttp.client import ClientSession

    from ..runner import Runner
    from ..runner import RunnerMethod
    from ....triton import Runner as TritonRunner

    P = t.ParamSpec("P")
    R = t.TypeVar("R")
else:
    P = t.TypeVar("P")

    tritongrpcclient = LazyLoader(
        "tritongrpcclient",
        globals(),
        "tritonclient.grpc.aio",
        exc_msg=TRITON_EXC_MSG,
    )
    tritonhttpclient = LazyLoader(
        "tritonhttpclient",
        globals(),
        "tritonclient.http.aio",
        exc_msg=TRITON_EXC_MSG,
    )

logger = logging.getLogger(__name__)


class RemoteRunnerClient(RunnerHandle):
    def __init__(self, runner: Runner):  # pylint: disable=super-init-not-called
        self._runner = runner
        self._conn: BaseConnector | None = None
        self._client_cache: ClientSession | None = None
        self._loop: asyncio.AbstractEventLoop | None = None
        self._addr: str | None = None
        self._semaphore = asyncio.Semaphore(
            BentoMLContainer.api_server_config.max_runner_connections.get()
        )

    @property
    def _remote_runner_server_map(self) -> dict[str, str]:
        return BentoMLContainer.remote_runner_mapping.get()

    @property
    def runner_timeout(self) -> int:
        "return the configured timeout for this runner."
        runner_cfg = BentoMLContainer.runners_config.get()
        if self._runner.name in runner_cfg:
            return runner_cfg[self._runner.name]["timeout"]
        else:
            return runner_cfg["timeout"]

    def _close_conn(self) -> None:
        if self._conn:
            self._conn.close()

    def _get_conn(self) -> BaseConnector:
        import aiohttp

        if (
            self._loop is None
            or self._conn is None
            or self._conn.closed
            or self._loop.is_closed()
        ):
            self._loop = asyncio.get_event_loop()  # get the loop lazily
            bind_uri = self._remote_runner_server_map[self._runner.name]
            parsed = urlparse(bind_uri)
            if parsed.scheme == "file":
                path = uri_to_path(bind_uri)
                self._conn = aiohttp.UnixConnector(
                    path=path,
                    loop=self._loop,
                    limit=800,  # TODO(jiang): make it configurable
                    keepalive_timeout=1800.0,
                )
                self._addr = "http://127.0.0.1:8000"  # addr doesn't matter with UDS
            elif parsed.scheme == "tcp":
                self._conn = aiohttp.TCPConnector(
                    loop=self._loop,
                    ssl=False,
                    limit=800,  # TODO(jiang): make it configurable
                    keepalive_timeout=1800.0,
                )
                self._addr = f"http://{parsed.netloc}"
            else:
                raise ValueError(f"Unsupported bind scheme: {parsed.scheme}") from None
        return self._conn

    @property
    def _client(self) -> ClientSession:
        import aiohttp

        if (
            self._loop is None
            or self._client_cache is None
            or self._client_cache.closed
            or self._loop.is_closed()
        ):
            from opentelemetry.instrumentation.aiohttp_client import create_trace_config

            def strip_query_params(url: yarl.URL) -> str:
                return str(url.with_query(None))

            jar = aiohttp.DummyCookieJar()
            timeout = aiohttp.ClientTimeout(total=self.runner_timeout)
            self._client_cache = aiohttp.ClientSession(
                trace_configs=[
                    create_trace_config(
                        # Remove all query params from the URL attribute on the span.
                        url_filter=strip_query_params,
                        tracer_provider=BentoMLContainer.tracer_provider.get(),
                    )
                ],
                connector=self._get_conn(),
                auto_decompress=False,
                cookie_jar=jar,
                connector_owner=False,
                timeout=timeout,
                loop=self._loop,
                trust_env=True,
            )
        return self._client_cache

    async def _reset_client(self):
        self._close_conn()
        if self._client_cache is not None:
            await self._client_cache.close()
            self._client_cache = None

    async def async_run_method(
        self,
        __bentoml_method: RunnerMethod[t.Any, P, R],
        *args: P.args,
        **kwargs: P.kwargs,
    ) -> R | tuple[R, ...]:
        import aiohttp

        from ...runner.container import AutoContainer

        inp_batch_dim = __bentoml_method.config.batch_dim[0]

        payload_params = Params[Payload](*args, **kwargs).map(
            functools.partial(AutoContainer.to_payload, batch_dim=inp_batch_dim)
        )

        headers = {
            "Bento-Name": component_context.bento_name,
            "Bento-Version": component_context.bento_version,
            "Runner-Name": self._runner.name,
            "Yatai-Bento-Deployment-Name": component_context.yatai_bento_deployment_name,
            "Yatai-Bento-Deployment-Namespace": component_context.yatai_bento_deployment_namespace,
        }
        total_args_num = len(args) + len(kwargs)
        headers["Args-Number"] = str(total_args_num)

        if total_args_num == 1:
            # FIXME: also considering kwargs
            if len(kwargs) == 1:
                kwarg_name = list(kwargs.keys())[0]
                headers["Kwarg-Name"] = kwarg_name
                payload = AutoContainer.to_payload(
                    kwargs[kwarg_name], batch_dim=inp_batch_dim
                )
            else:
                payload = AutoContainer.to_payload(args[0], batch_dim=inp_batch_dim)
            data = payload.data

            headers["Payload-Meta"] = json.dumps(payload.meta)
            headers["Payload-Container"] = payload.container
            headers["Batch-Size"] = str(payload.batch_size)

        else:
            payload_params = Params[Payload](*args, **kwargs).map(
                functools.partial(AutoContainer.to_payload, batch_dim=inp_batch_dim)
            )

            if __bentoml_method.config.batchable:
                if not payload_params.map(lambda i: i.batch_size).all_equal():
                    raise ValueError(
                        "All batchable arguments must have the same batch size."
                    )

            data = pickle.dumps(payload_params)  # FIXME: pickle inside pickle

        path = "" if __bentoml_method.name == "__call__" else __bentoml_method.name
        async with self._semaphore:
            try:
                async with self._client.post(
                    f"{self._addr}/{path}",
                    data=data,
                    headers=headers,
                ) as resp:
                    body = await resp.read()
            except aiohttp.ClientOSError as e:
                if os.getenv("BENTOML_RETRY_RUNNER_REQUESTS", "").lower() == "true":
                    try:
                        # most likely the TCP connection has been closed; retry after reconnecting
                        await self._reset_client()
                        async with self._client.post(
                            f"{self._addr}/{path}",
                            data=data,
                            headers=headers,
                        ) as resp:
                            body = await resp.read()
                    except aiohttp.ClientOSError:
                        raise RemoteException("Failed to connect to runner server.")
                else:
                    raise RemoteException("Failed to connect to runner server.") from e

        try:
            content_type = resp.headers["Content-Type"]
            assert content_type.lower().startswith("application/vnd.bentoml.")
        except (KeyError, AssertionError):
            raise RemoteException(
                f"An unexpected exception occurred in remote runner {self._runner.name}: [{resp.status}] {body.decode()}"
            ) from None

        if resp.status != 200:
            if resp.status == 503:
                raise ServiceUnavailable(body.decode()) from None
            if resp.status == 500:
                raise RemoteException(body.decode()) from None
            raise RemoteException(
                f"An exception occurred in remote runner {self._runner.name}: [{resp.status}] {body.decode()}"
            ) from None

        try:
            meta_header = resp.headers[PAYLOAD_META_HEADER]
        except KeyError:
            raise RemoteException(
                f"Bento payload decode error: {PAYLOAD_META_HEADER} header not set. An exception might have occurred in the remote server. [{resp.status}] {body.decode()}"
            ) from None

        if content_type == "application/vnd.bentoml.multiple_outputs":
            payloads = pickle.loads(body)
            return tuple(AutoContainer.from_payload(payload) for payload in payloads)

        container = content_type.strip("application/vnd.bentoml.")

        try:
            payload = Payload(
                data=body, meta=json.loads(meta_header), container=container
            )
        except JSONDecodeError:
            raise ValueError(f"Bento payload decode error: {meta_header}") from None

        return AutoContainer.from_payload(payload)

    def run_method(
        self,
        __bentoml_method: RunnerMethod[t.Any, P, R],
        *args: P.args,
        **kwargs: P.kwargs,
    ) -> R | tuple[R, ...]:
        import anyio

        return t.cast(
            "R | tuple[R, ...]",
            anyio.from_thread.run(
                functools.partial(self.async_run_method, **kwargs),
                __bentoml_method,
                *args,
            ),
        )

    async def is_ready(self, timeout: int) -> bool:
        import aiohttp

        # default kubernetes probe timeout is also 1s; see
        # https://kubernetes.io/docs/tasks/configure-pod-container/configure-liveness-readiness-startup-probes/#configure-probes
        aio_timeout = aiohttp.ClientTimeout(total=timeout)
        async with self._client.get(
            f"{self._addr}/readyz",
            headers={
                "Bento-Name": component_context.bento_name,
                "Bento-Version": component_context.bento_version,
                "Runner-Name": self._runner.name,
                "Yatai-Bento-Deployment-Name": component_context.yatai_bento_deployment_name,
                "Yatai-Bento-Deployment-Namespace": component_context.yatai_bento_deployment_namespace,
            },
            timeout=aio_timeout,
        ) as resp:
            return resp.status == 200

    def __del__(self) -> None:
        self._close_conn()


def handle_triton_exception(f: t.Callable[P, R]) -> t.Callable[..., R]:
    @functools.wraps(f)
    def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
        try:
            return f(*args, **kwargs)
        except tritongrpcclient.InferenceServerException:
            logger.error("Caught exception while sending payload to Triton:")
            raise

    return wrapper


# NOTE: to support files, consider using ModelInferStream via raw_bytes_contents
class TritonRunnerHandle(RunnerHandle):
    def __init__(self, runner: TritonRunner):
        self.runner = runner
        self._client_cache: t.Any = None
        self._grpc_client_cache: tritongrpcclient.InferenceServerClient | None = None
        self._http_client_cache: tritonhttpclient.InferenceServerClient | None = None
        self._use_http_client = self.runner.tritonserver_type == "http"

    async def is_ready(self, timeout: int) -> bool:
        logger.info("Waiting for Triton server to be ready...")
        start = time.time()
        while time.time() - start < timeout:
            try:
                if (
                    await self.client.is_server_ready()
                    and await self.client.is_server_live()
                ):
                    return True
                else:
                    await asyncio.sleep(1)
            except Exception:
                await asyncio.sleep(1)
        return False

    # keep a copy of all client methods to avoid getattr check.
    client_methods = [
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

    @property
    def http_client(self) -> tritonhttpclient.InferenceServerClient:
        from ...configuration import get_debug_mode

        if self._http_client_cache is None:
            # TODO: configuration customization
            try:
                self._http_client_cache = tritonhttpclient.InferenceServerClient(
                    url=BentoMLContainer.remote_runner_mapping.get()[self.runner.name],
                    verbose=get_debug_mode(),
                )
            except Exception:
                logger.error(
                    "Failed to instantiate Triton Inference Server client for '%s', see details:",
                    self.runner.name,
                )
                logger.error(traceback.format_exc())
                raise
        return self._http_client_cache

    @property
    def grpc_client(self) -> tritongrpcclient.InferenceServerClient:
        from ...configuration import get_debug_mode

        if self._grpc_client_cache is None:
            # TODO: configuration customization
            try:
                self._grpc_client_cache = tritongrpcclient.InferenceServerClient(
                    url=BentoMLContainer.remote_runner_mapping.get()[self.runner.name],
                    verbose=get_debug_mode(),
                )
            except Exception:
                logger.error(
                    "Failed to instantiate Triton Inference Server client for '%s', see details:",
                    self.runner.name,
                )
                logger.error(traceback.format_exc())
                raise
        return self._grpc_client_cache

    @property
    def client(
        self,
    ) -> (
        tritongrpcclient.InferenceServerClient | tritonhttpclient.InferenceServerClient
    ):
        if self._client_cache is None:
            self._client_cache = (
                self.http_client if self._use_http_client else self.grpc_client
            )

        return self._client_cache

    @handle_triton_exception
    async def async_run_method(
        self,
        __bentoml_method: RunnerMethod[t.Any, P, tritongrpcclient.InferResult],
        *args: P.args,
        **kwargs: P.kwargs,
    ) -> tritongrpcclient.InferResult | tritonhttpclient.InferResult:
        from ..container import AutoContainer

        assert (len(args) == 0) ^ (
            len(kwargs) == 0
        ), f"Inputs for model '{__bentoml_method.name}' can be given either as positional (args) or keyword arguments (kwargs), but not both. See https://github.com/triton-inference-server/server/blob/main/docs/user_guide/model_configuration.md#model-configuration"

        pass_args = args if len(args) > 0 else kwargs

        # return metadata of a given model
        if not self._use_http_client:
            metadata = await self.grpc_client.get_model_metadata(
                model_name=__bentoml_method.name, as_json=False
            )

            if len(metadata.inputs) != len(pass_args):
                raise ValueError(
                    f"Number of provided arguments ({len(metadata.inputs)}) does not match the number of inputs ({len(pass_args)})"
                )
            inputs: t.Sequence[t.Any] = metadata.inputs
            outputs: t.Sequence[t.Any] = [
                tritongrpcclient.InferRequestedOutput(output.name)
                for output in metadata.outputs
            ]
        else:
            metadata = await self.http_client.get_model_metadata(
                model_name=__bentoml_method.name
            )
            inputs = metadata["inputs"]

            if len(inputs) != len(pass_args):
                raise ValueError(
                    f"Number of provided arguments ({len(inputs)}) does not match the number of inputs ({len(pass_args)})"
                )
            outputs = [
                tritonhttpclient.InferRequestedOutput(output["name"])
                for output in t.cast("list[dict[str, t.Any]]", metadata["outputs"])
            ]

        param_cls = (
            Params[tritongrpcclient.InferInput]
            if not self._use_http_client
            else Params[tritonhttpclient.InferInput]
        )

        params = param_cls(*args, **kwargs).map_enumerate(
            functools.partial(
                AutoContainer.to_triton_payload, _use_http_client=self._use_http_client
            ),
            inputs,
        )
        return await self.client.infer(
            model_name=__bentoml_method.name,
            inputs=list(params.args) if len(args) > 0 else list(params.kwargs.values()),
            outputs=outputs,
        )

    @handle_triton_exception
    def run_method(
        self,
        __bentoml_method: RunnerMethod[t.Any, P, tritongrpcclient.InferResult],
        *args: P.args,
        **kwargs: P.kwargs,
    ) -> tritongrpcclient.InferResult | tritonhttpclient.InferResult:
        import anyio

        return anyio.from_thread.run(
            functools.partial(self.async_run_method, **kwargs),
            __bentoml_method,
            *args,
        )
