from typing import Dict, TYPE_CHECKING

from simple_di import Provide, inject

from bentoml._internal.configuration.containers import BentoMLContainer

if TYPE_CHECKING:
    from aiohttp import BaseConnector

    from .runner_transport import Transporter


class RunnerClient:
    def __init__(self, uds, timeout=None, transporter: "Transporter" = None):
        self._uds = uds
        self._conn = None
        self._client = None
        self._timeout = timeout
        from .runner_transport import PlasmaNdarrayTransporter

        self._transporter = transporter or PlasmaNdarrayTransporter()

    def _get_conn(self) -> "BaseConnector":
        import aiohttp

        if self._conn is None or self._conn.closed:
            self._conn = aiohttp.UnixConnector(
                path=self._uds,
            )
        return self._conn

    def _get_client(self):
        import aiohttp

        if self._client is None or self._client.closed:
            jar = aiohttp.DummyCookieJar()
            if self._timeout:
                timeout = aiohttp.ClientTimeout(total=self._timeout)
            else:
                timeout = None
            self._client = aiohttp.ClientSession(
                connector=self._get_conn(),
                auto_decompress=False,
                cookie_jar=jar,
                connector_owner=False,
                timeout=timeout,
            )
        return self._client

    async def async_run(self, *args, **kwargs):
        URL = "http://127.0.0.1:8000/run"
        payloads = {k: self._transporter.to_payload(v) for k, v in kwargs.items()}
        async with self._get_client() as client:
            async with client.post(URL, data=payloads) as resp:
                text = await resp.text()
        return self._transporter.from_payload(text)

    async def async_run_batch(self, *args, **kwargs):
        URL = "http://127.0.0.1:8000/run_batch"
        payloads = {k: self._transporter.to_payload(v) for k, v in kwargs.items()}
        async with self._get_client() as client:
            async with client.post(URL, data=payloads) as resp:
                text = await resp.text()
        return self._transporter.from_payload(text)


@inject
def get_runner_client(
    runner_name: str,
    uds_mapping: Dict[str, int] = Provide[BentoMLContainer.uds_mapping],
    timeout: int = Provide[BentoMLContainer.config.bento_server.timeout],
) -> RunnerClient:
    uds = uds_mapping.get(runner_name)
    return RunnerClient(uds, timeout=timeout)
