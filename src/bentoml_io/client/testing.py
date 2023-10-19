import typing as t

from ..server.service import Service
from .base import AbstractClient


class TestingClient(AbstractClient):
    def __init__(self, service: Service):
        self.servable = service.get_servable()
        for name in self.servable.__servable_methods__:
            setattr(self, name, getattr(self.servable, name))

    def call(self, name: str, *args: t.Any, **kwargs: t.Any) -> t.Any:
        if name not in self.servable.__servable_methods__:
            raise ValueError(f"Method {name} not found")
        return getattr(self.servable, name)(*args, **kwargs)
