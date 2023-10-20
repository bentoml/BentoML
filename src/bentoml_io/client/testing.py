from __future__ import annotations

import typing as t

from bentoml_io.client.local import LocalClient

if t.TYPE_CHECKING:
    from ..server.service import Service

T = t.TypeVar("T")


class TestingClient(LocalClient):
    def __init__(self, service: Service):
        super().__init__(service)
        for name in self.servable.__servable_methods__:
            setattr(self, name, getattr(self.servable, name))

    def call(self, name: str, *args: t.Any, **kwargs: t.Any) -> t.Any:
        if name not in self.servable.__servable_methods__:
            raise ValueError(f"Method {name} not found")
        return getattr(self.servable, name)(*args, **kwargs)
