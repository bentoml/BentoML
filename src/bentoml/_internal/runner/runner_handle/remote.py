from __future__ import annotations

import os
import json
import time
import pickle
import typing as t
import asyncio
import logging
import functools
from json.decoder import JSONDecodeError
from urllib.parse import urlparse

from . import RunnerHandle
from ..utils import Params
from ..utils import PAYLOAD_META_HEADER
from ...utils import LazyLoader
from ..runner import RunnerMethod
from ..runner import object_setattr
from ...context import component_context
from ..runnable import RunnableMethodConfig
from ..container import Payload
from ...utils.uri import uri_to_path
from ....exceptions import RemoteException
from ....exceptions import ServiceUnavailable
from ...configuration.containers import BentoMLContainer

if t.TYPE_CHECKING:
    import yarl
    import tritonclient.grpc.aio as tritongrpcclient
    from aiohttp import BaseConnector
    from aiohttp.client import ClientSession
    from tritonclient.grpc import service_pb2 as pb

    from ... import external_typing as ext
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
        exc_msg="tritonclient is required to use triton with BentoML. Install with 'pip install bentoml[triton]'.",
    )

logger = logging.getLogger(__name__)


class RemoteRunnerClient(RunnerHandle):
    def __init__(self, runner: Runner):  # pylint: disable=super-init-not-called
        self._runner = runner
        self._conn: BaseConnector | None = None
        self._client_cache: ClientSession | None = None
        self._loop: asyncio.AbstractEventLoop | None = None
        self._addr: str | None = None

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
                    verify_ssl=False,
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

        if __bentoml_method.config.batchable:
            if not payload_params.map(lambda i: i.batch_size).all_equal():
                raise ValueError(
                    "All batchable arguments must have the same batch size."
                ) from None

        path = "" if __bentoml_method.name == "__call__" else __bentoml_method.name
        try:
            async with self._client.post(
                f"{self._addr}/{path}",
                data=pickle.dumps(payload_params),  # FIXME: pickle inside pickle
                headers={
                    "Bento-Name": component_context.bento_name,
                    "Bento-Version": component_context.bento_version,
                    "Runner-Name": self._runner.name,
                    "Yatai-Bento-Deployment-Name": component_context.yatai_bento_deployment_name,
                    "Yatai-Bento-Deployment-Namespace": component_context.yatai_bento_deployment_namespace,
                },
            ) as resp:
                body = await resp.read()
        except aiohttp.ClientOSError as e:
            if os.getenv("BENTOML_RETRY_RUNNER_REQUESTS", "").lower() == "true":
                try:
                    # most likely the TCP connection has been closed; retry after reconnecting
                    await self._reset_client()
                    async with self._client.post(
                        f"{self._addr}/{path}",
                        data=pickle.dumps(
                            payload_params
                        ),  # FIXME: pickle inside pickle
                        headers={
                            "Bento-Name": component_context.bento_name,
                            "Bento-Version": component_context.bento_version,
                            "Runner-Name": self._runner.name,
                            "Yatai-Bento-Deployment-Name": component_context.yatai_bento_deployment_name,
                            "Yatai-Bento-Deployment-Namespace": component_context.yatai_bento_deployment_namespace,
                        },
                    ) as resp:
                        body = await resp.read()
                except aiohttp.ClientOSError as e:
                    raise RemoteException(f"Failed to connect to runner server.")
            else:
                raise RemoteException(f"Failed to connect to runner server.") from e

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
        except tritongrpcclient.InferenceServerException as err:
            logger.error("Caught exception while sending payload to Triton:")
            logger.error(err)
            raise err

    return wrapper


# NOTE: to support files, consider using ModelInferStream via raw_bytes_contents
class TritonRunnerHandle(RunnerHandle):
    def __init__(self, runner: TritonRunner):
        self.runner = runner
        self._client_cache: tritongrpcclient.InferenceServerClient | None = None
        self._loop: asyncio.AbstractEventLoop | None = None

    async def is_ready(self, timeout: int) -> bool:
        start = time.time()
        while time.time() - start < timeout:
            try:
                if (
                    await self._client.is_server_ready()
                    and await self._client.is_server_live()
                ):
                    return True
                else:
                    await asyncio.sleep(1)
            except Exception as err:
                logger.error("Caught exception while waiting Triton to be ready:")
                logger.error(err)
                await asyncio.sleep(1)
        return False

    @property
    def _url(self) -> str:
        try:
            return BentoMLContainer.remote_runner_mapping.get()[self.runner.name]
        except KeyError:
            raise ValueError(
                f"'{self.runner.name}' is not found in registered Triton runner mapping '{BentoMLContainer.remote_runner_mapping.get()}'"
            )

    @property
    def _client(self) -> tritongrpcclient.InferenceServerClient:
        from ...configuration import get_debug_mode

        if self._client_cache is None or self._loop is None or self._loop.is_closed():
            try:
                # TODO: configuration customization
                self._client_cache = tritongrpcclient.InferenceServerClient(
                    url=self._url, verbose=get_debug_mode()
                )
                self._loop = asyncio.get_event_loop()
            except Exception:
                import traceback

                logger.error(
                    "Failed to instantiate Triton Inference Server client for '%s', see details:",
                    self.runner.name,
                )
                logger.error(traceback.format_exc())
                raise
        return self._client_cache

    def __del__(self):
        if self._loop is not None:
            self._loop.call_soon_threadsafe(
                lambda: asyncio.ensure_future(self._client.close())
            )

    @handle_triton_exception
    async def get_model(
        self, model_name: str
    ) -> RunnerMethod[t.Any, P, tritongrpcclient.InferResult]:
        if not await self._client.is_model_ready(model_name):
            # model is not ready, try to load it
            logger.debug("model '%s' is not ready, loading.", model_name)
            await self._client.load_model(model_name)

        method = RunnerMethod[t.Any, P, tritongrpcclient.InferResult](
            runner=self.runner,
            name=model_name,
            # TODO: configuration for triton
            config=RunnableMethodConfig(batchable=False, batch_dim=(0, 0)),
            max_batch_size=0,
            max_latency_ms=10000,
        )

        # setattr for given models
        if model_name not in self.runner.__dict__:
            object_setattr(self.runner, model_name, method)

        return method

    @handle_triton_exception
    async def get_model_config(self, model_name: str) -> pb.ModelConfigResponse:
        if not await self._client.is_model_ready(model_name):
            await self.get_model(model_name)
        return t.cast(
            "pb.ModelConfigResponse", await self._client.get_model_config(model_name)
        )

    def __getattr__(self, item: str) -> t.Any:
        if item not in self.__dict__:
            try:
                func_or_attribute = getattr(self._client, item)
                if callable(func_or_attribute):
                    return handle_triton_exception(func_or_attribute)
                return None
            except AttributeError:
                pass
        return super().__getattribute__(item)

    @handle_triton_exception
    async def async_run_method(
        self,
        __bentoml_method: RunnerMethod[t.Any, P, tritongrpcclient.InferResult],
        *args: P.args,
        **kwargs: P.kwargs,
    ) -> tritongrpcclient.InferResult:
        from ..container import AutoContainer

        assert (len(args) == 0) ^ (
            len(kwargs) == 0
        ), f"Inputs for model '{__bentoml_method.name}' can be given either as positional (args) or keyword arguments (kwargs), but not both. See https://github.com/triton-inference-server/server/blob/main/docs/user_guide/model_configuration.md#model-configuration"

        model_name = __bentoml_method.name

        # return metadata of a given model
        model_metadata: pb.ModelMetadataResponse = (
            await self._client.get_model_metadata(model_name)
        )

        pass_args = args if len(args) > 0 else kwargs
        if len(model_metadata.inputs) != len(pass_args):
            raise ValueError(
                f"Number of provided arguments ({len(model_metadata.inputs)}) does not match the number of inputs ({len(pass_args)})"
            )

        input_params = Params["ext.NpNDArray"](*args, **kwargs).map(
            AutoContainer.to_triton_payload
        )

        outputs = [
            tritongrpcclient.InferRequestedOutput(output.name)
            for output in model_metadata.outputs
        ]
        inputs: list[tritongrpcclient.InferInput] = []

        if len(args) > 0:
            for (infer_input, arg) in zip(model_metadata.inputs, input_params.args):
                InferInput = tritongrpcclient.InferInput(
                    infer_input.name,
                    arg.shape,
                    tritongrpcclient.np_to_triton_dtype(arg.dtype),
                )
                InferInput.set_data_from_numpy(arg)
                inputs.append(InferInput)
        else:
            for infer_input in model_metadata.inputs:
                arg = input_params.kwargs[infer_input.name]
                inputs.append(
                    tritongrpcclient.InferInput(
                        infer_input.name,
                        arg.shape,
                        tritongrpcclient.np_to_triton_dtype(arg.dtype),
                    )
                )
                inputs[-1].set_data_from_numpy(arg)

        return await self._client.infer(
            model_name=model_name, inputs=inputs, outputs=outputs
        )

    @handle_triton_exception
    def run_method(
        self,
        __bentoml_method: RunnerMethod[t.Any, P, tritongrpcclient.InferResult],
        *args: P.args,
        **kwargs: P.kwargs,
    ) -> tritongrpcclient.InferResult:
        import anyio

        return anyio.from_thread.run(
            functools.partial(self.async_run_method, **kwargs),
            __bentoml_method,
            *args,
        )
