import json
import typing as t
from typing import TYPE_CHECKING
from json.decoder import JSONDecodeError

from simple_di import inject
from simple_di import Provide

from bentoml._internal.runner.container import Payload

from .runner import RunnerImpl
from ..runner.utils import Params
from ..runner.utils import PAYLOAD_META_HEADER
from ..runner.utils import payload_params_to_multipart
from ..configuration.containers import BentoServerContainer

if TYPE_CHECKING:  # pragma: no cover
    from aiohttp import BaseConnector
    from aiohttp.client import ClientSession


class RemoteRunnerClient(RunnerImpl):
    _conn: t.Optional["BaseConnector"] = None
    _client: t.Optional["ClientSession"] = None

    @inject
    def _get_conn(
        self,
        remote_runner_mapping: t.Dict[str, str] = Provide[
            BentoServerContainer.remote_runner_mapping
        ],
    ) -> "BaseConnector":
        import aiohttp

        uds: str = remote_runner_mapping[self._runner.name]
        if self._conn is None or self._conn.closed:
            self._conn = aiohttp.UnixConnector(path=uds)
        return self._conn

    @inject
    def _get_client(
        self,
        timeout_sec: t.Optional[float] = None,
    ) -> "ClientSession":
        import aiohttp

        if self._client is None or self._client.closed:
            jar = aiohttp.DummyCookieJar()
            if timeout_sec is not None:
                timeout = aiohttp.ClientTimeout(total=timeout_sec)
            else:
                DEFAULT_TIMEOUT = aiohttp.ClientTimeout(total=5 * 60)
                timeout = DEFAULT_TIMEOUT
            self._client = aiohttp.ClientSession(
                connector=self._get_conn(),
                auto_decompress=False,
                cookie_jar=jar,
                connector_owner=False,
                timeout=timeout,
            )
        return self._client

    async def _async_req(self, url: str, *args: t.Any, **kwargs: t.Any) -> t.Any:
        from ..runner.container import AutoContainer

        params = Params(*args, **kwargs).map(AutoContainer.single_to_payload)
        multipart = payload_params_to_multipart(params)
        client = self._get_client()
        async with client.post(url, data=multipart) as resp:
            body = await resp.read()
            meta_header = resp.headers[PAYLOAD_META_HEADER]
        try:
            payload = Payload(data=body, meta=json.loads(meta_header))
        except JSONDecodeError:
            raise ValueError(f"Bento payload decode error: {meta_header}")
        return AutoContainer.payload_to_single(payload)

    async def async_run(self, *args: t.Any, **kwargs: t.Any) -> t.Any:
        url = "http://127.0.0.1:8000/run"
        return await self._async_req(url, *args, **kwargs)

    async def async_run_batch(self, *args: t.Any, **kwargs: t.Any) -> t.Any:
        url = "http://127.0.0.1:8000/run_batch"
        return await self._async_req(url, *args, **kwargs)

    def run(self, *args: t.Any, **kwargs: t.Any) -> t.Any:
        import anyio

        return anyio.run(self.async_run, *args, **kwargs)

    def run_batch(self, *args: t.Any, **kwargs: t.Any) -> t.Any:
        import anyio

        return anyio.run(self.async_run_batch, *args, **kwargs)
