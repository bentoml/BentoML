from __future__ import annotations

import os
import json
import pickle
import typing as t
import asyncio
import functools
from typing import TYPE_CHECKING
from json.decoder import JSONDecodeError
from urllib.parse import urlparse

from . import RunnerHandle
from ...context import component_context
from ..container import Payload
from ...utils.uri import uri_to_path
from ....exceptions import RemoteException
from ....exceptions import ServiceUnavailable
from ...runner.utils import Params
from ...runner.utils import PAYLOAD_META_HEADER
from ...configuration.containers import BentoMLContainer

if TYPE_CHECKING:
    import yarl
    from aiohttp import BaseConnector
    from aiohttp.client import ClientSession

    from ..runner import Runner
    from ..runner import RunnerMethod

    P = t.ParamSpec("P")
    R = t.TypeVar("R")


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
    def _client(
        self,
    ) -> ClientSession:
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
    ) -> R:
        import anyio

        return t.cast(
            "R",
            anyio.from_thread.run(
                self.async_run_method,
                __bentoml_method,
                *args,
                **kwargs,
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
