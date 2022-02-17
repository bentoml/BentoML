import json
import typing as t
import asyncio
from typing import TYPE_CHECKING
from json.decoder import JSONDecodeError
from urllib.parse import urlparse

from simple_di import inject
from simple_di import Provide

from .runner import RunnerImpl
from .container import Payload
from ..utils.uri import uri_to_path
from ..runner.utils import Params
from ..runner.utils import PAYLOAD_META_HEADER
from ..runner.utils import payload_params_to_multipart
from ..configuration.containers import DeploymentContainer

if TYPE_CHECKING:  # pragma: no cover
    from aiohttp import BaseConnector
    from aiohttp.client import ClientSession


class RemoteRunnerClient(RunnerImpl):
    _conn: t.Optional["BaseConnector"] = None
    _client: t.Optional["ClientSession"] = None
    _loop: t.Optional[asyncio.AbstractEventLoop] = None
    _addr: t.Optional[str] = None

    def shutdown(self) -> None:
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
            self._loop = asyncio.get_event_loop()
            bind_uri = remote_runner_mapping[self._runner.name]
            parsed = urlparse(bind_uri)
            if parsed.scheme == "file":
                path = uri_to_path(bind_uri)
                self._conn = aiohttp.UnixConnector(
                    path=path,
                    loop=self._loop,
                    limit=self._runner.batch_options.max_batch_size * 2,
                    keepalive_timeout=self._runner.batch_options.max_latency_ms
                    * 1000
                    * 10,
                )
                self._addr = "http://127.0.0.1:8000"  # addr doesn't matter with UDS
            elif parsed.scheme == "tcp":
                self._conn = aiohttp.TCPConnector(
                    loop=self._loop,
                    limit=self._runner.batch_options.max_batch_size * 2,
                    verify_ssl=False,
                    keepalive_timeout=self._runner.batch_options.max_latency_ms
                    * 1000
                    * 10,
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
                        url_filter=strip_query_params,
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
            raise ValueError(
                f"Bento payload decode error: {PAYLOAD_META_HEADER} not exist. "
                "An exception might have occurred in the upstream server."
                f"[{resp.status}] {body.decode()}"
            )

        try:
            payload = Payload(data=body, meta=json.loads(meta_header))
        except JSONDecodeError:
            raise ValueError(f"Bento payload decode error: {meta_header}")

        return AutoContainer.payload_to_single(payload)

    async def async_run(self, *args: t.Any, **kwargs: t.Any) -> t.Any:
        return await self._async_req("run", *args, **kwargs)

    async def async_run_batch(self, *args: t.Any, **kwargs: t.Any) -> t.Any:
        return await self._async_req("run_batch", *args, **kwargs)

    def run(self, *args: t.Any, **kwargs: t.Any) -> t.Any:
        import anyio

        return anyio.from_thread.run(self.async_run, *args, **kwargs)

    def run_batch(self, *args: t.Any, **kwargs: t.Any) -> t.Any:
        import anyio

        return anyio.from_thread.run(self.async_run_batch, *args, **kwargs)

    def __del__(self) -> None:
        self.shutdown()
