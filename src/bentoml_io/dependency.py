from __future__ import annotations

import typing as t

import attrs
from simple_di import Provide
from simple_di import inject

from bentoml._internal.configuration.containers import BentoMLContainer

from .factory import Service

T = t.TypeVar("T")


_dependent_cache: dict[str, t.Any] = {}


def get_cache_key(service: Service[t.Any]) -> str:
    return service.name


@inject
def _get_dependency(
    service: Service[T],
    runner_mapping: dict[str, str] = Provide[BentoMLContainer.remote_runner_mapping],
) -> T:
    from .client.proxy import RemoteProxy

    key = get_cache_key(service)
    if key not in _dependent_cache:
        if key in runner_mapping:
            inst = RemoteProxy(runner_mapping[key], service=service).as_service()
        else:
            inst = service.inner()
        _dependent_cache[key] = inst
    return _dependent_cache[key]


@attrs.frozen
class Dependency(t.Generic[T]):
    on: Service[T]

    @t.overload
    def __get__(self, instance: None, owner: t.Any) -> Dependency[T]:
        ...

    @t.overload
    def __get__(self, instance: t.Any, owner: t.Any) -> T:
        ...

    def __get__(self, instance: t.Any, owner: t.Any) -> Dependency[T] | T:
        if instance is None:
            return self
        return _get_dependency(self.on)

    def __getattr__(self, name: str) -> t.Any:
        raise RuntimeError("Dependancy must be accessed as a class attribute")


def depends(on: Service[T]) -> Dependency[T]:
    if not isinstance(on, Service):
        raise TypeError(
            "depends() expects a class decorated with @bentoml_io.service()"
        )
    return Dependency(on)
