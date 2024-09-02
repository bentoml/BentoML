from __future__ import annotations

from typing import TYPE_CHECKING
from typing import Any

import attrs

from ..tasks import ResultStatus
from .base import map_exception

if TYPE_CHECKING:
    from .http import AsyncHTTPClient
    from .http import ClientEndpoint
    from .http import SyncHTTPClient


@attrs.frozen
class Task:
    id: str
    endpoint: ClientEndpoint = attrs.field(repr=False)
    client: SyncHTTPClient = attrs.field(repr=False)

    def get_status(self) -> ResultStatus:
        resp = self.client.request(
            "GET", f"{self.endpoint.route}/status", params={"task_id": self.id}
        )
        if resp.is_error:
            resp.read()
            raise map_exception(resp)
        data = resp.json()
        return ResultStatus(data["status"])

    def cancel(self) -> None:
        resp = self.client.request(
            "PUT", f"{self.endpoint.route}/cancel", params={"task_id": self.id}
        )
        if resp.is_error:
            resp.read()
            raise map_exception(resp)

    def get(self) -> Any:
        resp = self.client.request(
            "GET", f"{self.endpoint.route}/get", params={"task_id": self.id}
        )
        if resp.is_error:
            resp.read()
            raise map_exception(resp)
        if (
            self.endpoint.output.get("type") == "file"
            and self.client.media_type == "application/json"
        ):
            return self.client._parse_file_response(self.endpoint, resp)
        else:
            return self.client._parse_response(self.endpoint, resp)

    def retry(self) -> Task:
        resp = self.client.request(
            "POST", f"{self.endpoint.route}/retry", params={"task_id": self.id}
        )
        if resp.is_error:
            resp.read()
            raise map_exception(resp)
        data = resp.json()
        return Task(data["task_id"], self.endpoint, self.client)


@attrs.frozen
class AsyncTask:
    id: str
    endpoint: ClientEndpoint = attrs.field(repr=False)
    client: AsyncHTTPClient = attrs.field(repr=False)

    async def get_status(self) -> ResultStatus:
        resp = await self.client.request(
            "GET", f"{self.endpoint.route}/status", params={"task_id": self.id}
        )
        if resp.is_error:
            await resp.aread()
            raise map_exception(resp)
        data = resp.json()
        return ResultStatus(data["status"])

    async def cancel(self) -> None:
        resp = await self.client.request(
            "PUT", f"{self.endpoint.route}/cancel", params={"task_id": self.id}
        )
        if resp.is_error:
            await resp.aread()
            raise map_exception(resp)

    async def get(self) -> Any:
        resp = await self.client.request(
            "GET", f"{self.endpoint.route}/get", params={"task_id": self.id}
        )
        if resp.is_error:
            await resp.aread()
            raise map_exception(resp)
        if (
            self.endpoint.output.get("type") == "file"
            and self.client.media_type == "application/json"
        ):
            return await self.client._parse_file_response(self.endpoint, resp)
        else:
            return await self.client._parse_response(self.endpoint, resp)

    async def retry(self) -> AsyncTask:
        resp = await self.client.request(
            "POST", f"{self.endpoint.route}/retry", params={"task_id": self.id}
        )
        if resp.is_error:
            await resp.aread()
            raise map_exception(resp)
        data = resp.json()
        return AsyncTask(data["task_id"], self.endpoint, self.client)
