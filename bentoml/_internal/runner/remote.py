import json
import typing as t
import asyncio
from typing import TYPE_CHECKING
from json.decoder import JSONDecodeError
from urllib.parse import urlparse

import attr
from simple_di import inject
from simple_di import Provide

from .container import Payload
from ..utils.uri import uri_to_path
from ...exceptions import RemoteException
from ..runner.utils import Params
from ..runner.utils import PAYLOAD_META_HEADER
from ..runner.utils import payload_params_to_multipart
from ..configuration.containers import DeploymentContainer

if TYPE_CHECKING:  # pragma: no cover
    from aiohttp import BaseConnector
    from aiohttp.client import ClientSession

    from .runner import Runner


@attr.define
class RemoteRunnerClient:
    _runner: "Runner" = attr.field()
    _conn: t.Optional["BaseConnector"] = attr.field(init=False, default=None)
    _client: t.Optional["ClientSession"] = attr.field(init=False, default=None)
    _loop: t.Optional[asyncio.AbstractEventLoop] = attr.field(init=False, default=None)
    _addr: t.Optional[str] = attr.field(init=False, default=None)

    def _close_conn(self) -> None:
        if self._conn:
            self._conn.close()

    @inject
    def _get_conn(
        self,
        remote_runner_mapping: t.Dict[str, str] = Provide[
            DeploymentContainer.remote_runner_mapping
        ],
    ) -> "BaseConnector":
        import aiohttp

        if (
            self._loop is None
            or self._conn is None
            or self._conn.closed
            or self._loop.is_closed()
        ):
            self._loop = asyncio.get_event_loop()  # get the loop lazily
            bind_uri = remote_runner_mapping[self._runner.name]
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

    @inject
    def _get_client(
        self,
        timeout_sec: t.Optional[float] = None,
    ) -> "ClientSession":
        import aiohttp

        if (
            self._loop is None
            or self._client is None
            or self._client.closed
            or self._loop.is_closed()
        ):
            import yarl
            from opentelemetry.instrumentation.aiohttp_client import create_trace_config

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

    async def _async_req(self, path: str, *args: t.Any, **kwargs: t.Any) -> t.Any:
        from ..runner.container import AutoContainer

        params = Params(*args, **kwargs).map(AutoContainer.single_to_payload)
        multipart = payload_params_to_multipart(params)
        client = self._get_client()
        async with client.post(f"{self._addr}/{path}", data=multipart) as resp:
            body = await resp.read()
        try:
            meta_header = resp.headers[PAYLOAD_META_HEADER]
        except KeyError:
            raise RemoteException(
                f"Bento payload decode error: {PAYLOAD_META_HEADER} not exist. "
                "An exception might have occurred in the upstream server."
                f"[{resp.status}] {body.decode()}"
            ) from None

        try:
            payload = Payload(data=body, meta=json.loads(meta_header))
        except JSONDecodeError:
            raise ValueError(f"Bento payload decode error: {meta_header}")

        return AutoContainer.payload_to_single(payload)

    async def async_run_method(
        self,
        method_name: str,
        *args: t.Any,
        **kwargs: t.Any,
    ) -> t.Any:
        from ..runner.container import AutoContainer

        params = Params(*args, **kwargs).map(AutoContainer.single_to_payload)
        multipart = payload_params_to_multipart(params)
        client = self._get_client()
        async with client.post(f"{self._addr}/{method_name}", data=multipart) as resp:
            body = await resp.read()
        try:
            meta_header = resp.headers[PAYLOAD_META_HEADER]
        except KeyError:
            raise RemoteException(
                f"Bento payload decode error: {PAYLOAD_META_HEADER} not exist. "
                "An exception might have occurred in the remote server."
                f"[{resp.status}] {body.decode()}"
            ) from None

        try:
            payload = Payload(data=body, meta=json.loads(meta_header))
        except JSONDecodeError:
            raise ValueError(f"Bento payload decode error: {meta_header}")

        return AutoContainer.payload_to_single(payload)

    def run_method(
        self,
        method_name: str,
        *args: t.Any,
        **kwargs: t.Any,
    ) -> t.Any:
        import anyio

        return anyio.from_thread.run(
            self.async_run_method,
            method_name,
            *args,
            **kwargs,
        )

    def __del__(self) -> None:
        self._close_conn()
