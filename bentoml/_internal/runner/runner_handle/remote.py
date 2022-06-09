from __future__ import annotations

import json
import typing as t
import asyncio
import functools
from typing import TYPE_CHECKING
from json.decoder import JSONDecodeError
from urllib.parse import urlparse

from . import RunnerHandle
from ..container import Payload
from ...utils.uri import uri_to_path
from ....exceptions import RemoteException
from ...runner.utils import Params
from ...runner.utils import PAYLOAD_META_HEADER
from ...runner.utils import payload_params_to_multipart
from ...configuration.containers import DeploymentContainer

if TYPE_CHECKING:  # pragma: no cover
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
        self._client: ClientSession | None = None
        self._loop: asyncio.AbstractEventLoop | None = None
        self._addr: str | None = None

    @property
    def _remote_runner_server_map(self) -> dict[str, str]:
        return DeploymentContainer.remote_runner_mapping.get()

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
                raise ValueError(f"Unsupported bind scheme: {parsed.scheme}")
        return self._conn

    def _get_client(
        self,
        timeout_sec: float | None = None,
    ) -> ClientSession:
        import aiohttp

        if (
            self._loop is None
            or self._client is None
            or self._client.closed
            or self._loop.is_closed()
        ):
            import yarl
            from opentelemetry.instrumentation.aiohttp_client import (
                create_trace_config,  # type: ignore (missing type stubs)
            )

            def strip_query_params(url: yarl.URL) -> str:
                return str(url.with_query(None))

            jar = aiohttp.DummyCookieJar()
            if timeout_sec is not None:
                timeout = aiohttp.ClientTimeout(total=timeout_sec)
            else:
                DEFAULT_TIMEOUT = aiohttp.ClientTimeout(total=5 * 60)
                timeout = DEFAULT_TIMEOUT
            self._client = aiohttp.ClientSession(
                trace_configs=[
                    create_trace_config(
                        # Remove all query params from the URL attribute on the span.
                        url_filter=strip_query_params,  # type: ignore
                        tracer_provider=DeploymentContainer.tracer_provider.get(),
                    )
                ],
                connector=self._get_conn(),
                auto_decompress=False,
                cookie_jar=jar,
                connector_owner=False,
                timeout=timeout,
                loop=self._loop,
            )
        return self._client

    async def async_run_method(
        self,
        __bentoml_method: RunnerMethod[t.Any, P, R],
        *args: P.args,
        **kwargs: P.kwargs,
    ) -> R:
        from ...runner.container import AutoContainer

        # TODO: validate call signature

        inp_batch_dim = __bentoml_method.config.batch_dim[0]

        payload_params = Params[Payload](*args, **kwargs).map(
            functools.partial(AutoContainer.to_payload, batch_dim=inp_batch_dim)
        )

        if __bentoml_method.config.batchable:
            if not payload_params.map(lambda i: i.batch_size).all_equal():
                raise ValueError(
                    "All batchable arguments must have the same batch size."
                )

        multipart = payload_params_to_multipart(payload_params)
        client = self._get_client()
        path = "" if __bentoml_method.name == "__call__" else __bentoml_method.name
        async with client.post(
            f"{self._addr}/{path}",
            data=multipart,
        ) as resp:
            body = await resp.read()

        if resp.status != 200:
            raise RemoteException(
                f"An exception occurred in remote runner {self._runner.name}: [{resp.status}] {body.decode()}"
            )

        try:
            meta_header = resp.headers[PAYLOAD_META_HEADER]
        except KeyError:
            raise RemoteException(
                f"Bento payload decode error: {PAYLOAD_META_HEADER} header not set. "
                "An exception might have occurred in the remote server."
                f"[{resp.status}] {body.decode()}"
            ) from None

        try:
            content_type = resp.headers["Content-Type"]
        except KeyError:
            raise RemoteException(
                f"Bento payload decode error: Content-Type header not set. "
                "An exception might have occurred in the remote server."
                f"[{resp.status}] {body.decode()}"
            ) from None

        if not content_type.lower().startswith("application/vnd.bentoml."):
            raise RemoteException(
                f"Bento payload decode error: invalid Content-Type '{content_type}'."
            )

        container = content_type.strip("application/vnd.bentoml.")

        try:
            payload = Payload(
                data=body, meta=json.loads(meta_header), container=container
            )
        except JSONDecodeError:
            raise ValueError(f"Bento payload decode error: {meta_header}")

        return AutoContainer.from_payload(payload)

    def run_method(
        self,
        __bentoml_method: RunnerMethod[t.Any, P, R],
        *args: P.args,
        **kwargs: P.kwargs,
    ) -> R:
        import anyio

        return anyio.from_thread.run(  # type: ignore (pyright cannot infer the return type)
            self.async_run_method,
            __bentoml_method,
            *args,
            **kwargs,
        )

    def __del__(self) -> None:
        self._close_conn()
