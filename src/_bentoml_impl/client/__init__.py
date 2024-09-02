from __future__ import annotations

from .base import AbstractClient
from .http import AsyncHTTPClient as _AsyncHTTPClient
from .http import HTTPClient
from .http import SyncHTTPClient as _SyncHTTPClient
from .proxy import RemoteProxy
from .task import AsyncTask
from .task import Task

__all__ = [
    "AsyncHTTPClient",
    "SyncHTTPClient",
    "HTTPClient",
    "AbstractClient",
    "RemoteProxy",
    "Task",
    "AsyncTask",
]


class SyncHTTPClient(_SyncHTTPClient):
    """A synchronous client for BentoML service.

    Args:
        url (str): URL of the BentoML service.
        token (str, optional): Authentication token. Defaults to None.
        timeout (float, optional): Timeout for the client. Defaults to 30.
        server_ready_timeout (float, optional): Timeout for the server to be ready.
            If not provided, it will use the same value as `timeout`.

    Example::

        with SyncHTTPClient("http://localhost:3000") as client:
            resp = client.call("classify", input_series=[[1,2,3,4]])
            assert resp == [0]
            # Or using named method directly
            resp = client.classify(input_series=[[1,2,3,4]])
            assert resp == [0]
    """

    def __init__(
        self,
        url: str,
        *,
        token: str | None = None,
        timeout: float = 30,
        server_ready_timeout: float | None = None,
    ) -> None:
        super().__init__(
            url, token=token, timeout=timeout, server_ready_timeout=server_ready_timeout
        )


class AsyncHTTPClient(_AsyncHTTPClient):
    """An asynchronous client for BentoML service.

    Args:
        url (str): URL of the BentoML service.
        token (str, optional): Authentication token. Defaults to None.
        timeout (float, optional): Timeout for the client. Defaults to 30.
        server_ready_timeout (float, optional): Timeout for the server to be ready.
            If not provided, it will use the same value as `timeout`.

    Example::

        async with AsyncHTTPClient("http://localhost:3000") as client:
            resp = await client.call("classify", input_series=[[1,2,3,4]])
            assert resp == [0]
            # Or using named method directly
            resp = await client.classify(input_series=[[1,2,3,4]])
            assert resp == [0]

            # Streaming
            resp = client.stream(prompt="hello")
            async for data in resp:
                print(data)
    """

    def __init__(
        self,
        url: str,
        *,
        token: str | None = None,
        timeout: float = 30,
        server_ready_timeout: float | None = None,
    ) -> None:
        super().__init__(
            url, token=token, timeout=timeout, server_ready_timeout=server_ready_timeout
        )
