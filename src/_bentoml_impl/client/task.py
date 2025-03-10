from __future__ import annotations

from typing import TYPE_CHECKING
from typing import Any

import attrs

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
        return self.client._get_task_status(self.endpoint, self.id)

    def cancel(self) -> None:
        return self.client._cancel_task(self.endpoint, self.id)

    def get(self) -> Any:
        return self.client._get_task_result(self.endpoint, self.id)

    def retry(self) -> Task:
        return self.client._retry_task(self.endpoint, self.id)


@attrs.frozen
class AsyncTask:
    id: str
    endpoint: ClientEndpoint = attrs.field(repr=False)
    client: AsyncHTTPClient = attrs.field(repr=False)

    async def get_status(self) -> ResultStatus:
        return await self.client._get_task_status(self.endpoint, self.id)

    async def cancel(self) -> None:
        return await self.client._cancel_task(self.endpoint, self.id)

    async def get(self) -> Any:
        return await self.client._get_task_result(self.endpoint, self.id)

    async def retry(self) -> AsyncTask:
        return await self.client._retry_task(self.endpoint, self.id)
