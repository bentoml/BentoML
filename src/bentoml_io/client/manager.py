from __future__ import annotations

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

    def sync_client(self, name_or_class: str | type[Servable]) -> AbstractClient:
        cache = self._sync_clients
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
                client = SyncHTTPClient(
                    self._runner_map[name], servable=dep.servable_cls
                )
            else:
                client = SyncLocalClient(dep)
            cache[name] = client
        return cache[name]

    def async_client(self, name_or_class: str | type[Servable]) -> AbstractClient:
        cache = self._async_clients
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
                client = AsyncHTTPClient(
                    self._runner_map[name], servable=dep.servable_cls
                )
            else:
                client = AsyncLocalClient(dep)
            cache[name] = client
        return cache[name]

    async def cleanup(self) -> None:
        for client in self._async_clients.values():
            await client.__aexit__(None, None, None)
        for client in self._sync_clients.values():
            client.__exit__(None, None, None)
