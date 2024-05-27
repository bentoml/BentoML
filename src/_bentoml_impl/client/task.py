from __future__ import annotations

from http import HTTPStatus
from typing import TYPE_CHECKING
from typing import Any

import attrs

from bentoml.exceptions import BentoMLException

from ..tasks import ResultStatus

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
            raise BentoMLException(
                f"Error making request: {resp.status_code}: {resp.text}",
                error_code=HTTPStatus(resp.status_code),
            )
        data = resp.json()
        return ResultStatus(data["status"])

    def get(self) -> Any:
        resp = self.client.request(
            "GET", f"{self.endpoint.route}/get", params={"task_id": self.id}
        )
        if resp.is_error:
            resp.read()
            raise BentoMLException(
                f"Error making request: {resp.status_code}: {resp.text}",
                error_code=HTTPStatus(resp.status_code),
            )
        if (
            self.endpoint.output.get("type") == "file"
            and self.client.media_type == "application/json"
        ):
            return self.client._parse_file_response(self.endpoint, resp)
        else:
            return self.client._parse_response(self.endpoint, resp)


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
            raise BentoMLException(
                f"Error making request: {resp.status_code}: {resp.text}",
                error_code=HTTPStatus(resp.status_code),
            )
        data = resp.json()
        return ResultStatus(data["status"])

    async def get(self) -> Any:
        resp = await self.client.request(
            "GET", f"{self.endpoint.route}/get", params={"task_id": self.id}
        )
        if resp.is_error:
            await resp.aread()
            raise BentoMLException(
                f"Error making request: {resp.status_code}: {resp.text}",
                error_code=HTTPStatus(resp.status_code),
            )
        if (
            self.endpoint.output.get("type") == "file"
            and self.client.media_type == "application/json"
        ):
            return await self.client._parse_file_response(self.endpoint, resp)
        else:
            return await self.client._parse_response(self.endpoint, resp)
