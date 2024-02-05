from __future__ import annotations

import asyncio
import typing as t

import attrs
from simple_di import Provide
from simple_di import inject

from bentoml._internal.configuration.containers import BentoMLContainer

from .factory import Service

T = t.TypeVar("T")


_dependent_cache: dict[str, t.Any] = {}


async def cleanup() -> None:
    from _bentoml_impl.client.proxy import RemoteProxy

    coros: list[t.Coroutine[t.Any, t.Any, None]] = []
    for svc in _dependent_cache.values():
        if isinstance(svc, RemoteProxy):
            coros.append(svc.close())
    await asyncio.gather(*coros)
    _dependent_cache.clear()


@attrs.frozen
class Dependency(t.Generic[T]):
    on: Service[T]

    def cache_key(self) -> str:
        return self.on.name

    @inject
    def get(
        self,
        runner_mapping: dict[str, str] = Provide[
            BentoMLContainer.remote_runner_mapping
        ],
    ) -> T:
        from _bentoml_impl.client.proxy import RemoteProxy

        key = self.on.name
        if key not in _dependent_cache:
            if key in runner_mapping:
                inst = RemoteProxy(runner_mapping[key], service=self.on).as_service()
            else:
                inst = self.on()
            _dependent_cache[key] = inst
        return _dependent_cache[key]

    @t.overload
    def __get__(self, instance: None, owner: t.Any) -> Dependency[T]: ...

    @t.overload
    def __get__(self, instance: t.Any, owner: t.Any) -> T: ...

    def __get__(self, instance: t.Any, owner: t.Any) -> Dependency[T] | T:
        if instance is None:
            return self
        return self.get()

    def __getattr__(self, name: str) -> t.Any:
        raise RuntimeError("Dependancy must be accessed as a class attribute")


def depends(on: Service[T]) -> Dependency[T]:
    if not isinstance(on, Service):
        raise TypeError("depends() expects a class decorated with @bentoml.service()")
    return Dependency(on)
