from __future__ import annotations

import inspect
import typing as t

from simple_di import Provide
from simple_di import inject

from bentoml._internal.container import BentoMLContainer

from .base import AbstractClient
from .http import AsyncHTTPClient
from .http import SyncHTTPClient
from .local import AsyncLocalClient
from .local import SyncLocalClient

if t.TYPE_CHECKING:
    from ..servable import Servable
    from ..server import Service


class ClientManager:
    @inject
    def __init__(
        self,
        service: Service,
        runner_map: dict[str, str] = Provide[BentoMLContainer.remote_runner_mapping],
    ) -> None:
        self.service = service
        self._runner_map = runner_map
        self._sync_clients: dict[str, AbstractClient] = {}
        self._async_clients: dict[str, AbstractClient] = {}

    def get_client(self, name_or_class: str | type[Servable]) -> AbstractClient:
        caller_frame = inspect.currentframe().f_back  # type: ignore
        assert caller_frame is not None
        is_async = bool(
            caller_frame.f_code.co_flags & inspect.CO_COROUTINE
            or caller_frame.f_code.co_flags & inspect.CO_ASYNC_GENERATOR
        )
        cache = self._async_clients if is_async else self._sync_clients
        name = name_or_class if isinstance(name_or_class, str) else name_or_class.name
        if name not in cache:
            dep = next(
                (dep for dep in self.service.dependencies if dep.name == name), None
            )
            if dep is None:
                raise ValueError(
                    f"Dependency service {name} not found, please specify it in dependencies list"
                )
            if name in self._runner_map:
                client_cls = AsyncHTTPClient if is_async else SyncHTTPClient
                client = client_cls(self._runner_map[name], servable=dep.servable_cls)
            else:
                client_cls = AsyncLocalClient if is_async else SyncLocalClient
                client = client_cls(dep)
            cache[name] = client
        return cache[name]

    async def cleanup(self) -> None:
        for client in self._async_clients.values():
            await client.__aexit__(None, None, None)
        for client in self._sync_clients.values():
            await client.__aexit__(None, None, None)
