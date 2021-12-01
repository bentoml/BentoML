from typing import Dict, TYPE_CHECKING

from simple_di import inject, Provide

from ..runner.utils import Params
from ..configuration.containers import BentoServerContainer

if TYPE_CHECKING:
    from aiohttp import BaseConnector


class RunnerClient:
    def __init__(self, uds, timeout=None):
        self._uds = uds
        self._conn = None
        self._client = None
        self._timeout = timeout

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
        from ..runner.container import AutoContainer

        URL = "http://127.0.0.1:8000/run"
        param = Params(*args, **kwargs).map(AutoContainer.single_to_payload)
        payloads = param.to_dict()  # TODO(jiang): multipart
        async with self._get_client() as client:
            async with client.post(URL, data=payloads) as resp:
                text = await resp.text()
        return AutoContainer.payload_to_single(text)  # TODO

    async def async_run_batch(self, *args, **kwargs):
        from ..runner.container import AutoContainer

        URL = "http://127.0.0.1:8000/run_batch"
        param = Params(*args, **kwargs).map(AutoContainer.batch_to_payload)
        payloads = param.to_dict()
        async with self._get_client() as client:
            async with client.post(URL, data=payloads) as resp:
                text = await resp.text()
        return AutoContainer.payload_to_batch(text)  # TODO


@inject
def get_runner_client(
    runner_name: str,
    remote_runner_mapping: Dict[str, int] = Provide[
        BentoServerContainer.remote_runner_mapping
    ],
    timeout: int = Provide[BentoServerContainer.config.timeout],
) -> RunnerClient:
    uds_fd = remote_runner_mapping.get(runner_name)
    return RunnerClient(uds_fd, timeout=timeout)
