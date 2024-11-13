from __future__ import annotations

import asyncio
import typing as t

import attrs
from simple_di import Provide
from simple_di import inject

from bentoml._internal.configuration.containers import BentoMLContainer
from bentoml.exceptions import BentoMLException

from .factory import Service

if t.TYPE_CHECKING:
    from _bentoml_impl.client.proxy import RemoteProxy

T = t.TypeVar("T")


_dependencies: list[Dependency[t.Any]] = []


async def cleanup() -> None:
    tasks = [dep.close() for dep in _dependencies]
    await asyncio.gather(*tasks)
    _dependencies.clear()


@attrs.define
class Dependency(t.Generic[T]):
    on: Service[T] | None = None
    deployment: str | None = None
    cluster: str | None = None
    url: str | None = None
    _resolved: t.Any = attrs.field(default=None, init=False)

    @t.overload
    def get(self: Dependency[None]) -> RemoteProxy[t.Any]: ...

    @t.overload
    def get(self: Dependency[T]) -> T: ...

    @inject
    def get(
        self: Dependency[T],
        *,
        runner_mapping: dict[str, str] = Provide[
            BentoMLContainer.remote_runner_mapping
        ],
    ) -> T | RemoteProxy[t.Any]:
        from _bentoml_impl.client.proxy import RemoteProxy

        media_type = "application/json"
        if self.deployment and self.url:
            raise BentoMLException("Cannot specify both deployment and url")
        if self.deployment:
            client = BentoMLContainer.rest_api_client.get()
            deployment = client.v2.get_deployment(self.deployment, self.cluster)
            try:
                self.url = deployment.urls[0]
            except IndexError:
                raise BentoMLException(
                    f"Deployment {self.deployment} does not have any URLs"
                )
        elif not self.url:
            if self.on is None:
                raise BentoMLException("Must specify one of on, deployment or url")
            if (key := self.on.name) in runner_mapping:
                self.url = runner_mapping[key]
                media_type = "application/vnd.bentoml+pickle"
            else:
                return self.on()

        return RemoteProxy(
            self.url, service=self.on, media_type=media_type
        ).as_service()

    @t.overload
    def __get__(self, instance: None, owner: t.Any) -> t.Self: ...

    @t.overload
    def __get__(
        self: Dependency[None], instance: t.Any, owner: t.Any
    ) -> RemoteProxy[t.Any]: ...

    @t.overload
    def __get__(self: Dependency[T], instance: t.Any, owner: t.Any) -> T: ...

    def __get__(
        self: Dependency[T], instance: t.Any, owner: t.Any
    ) -> Dependency[T] | RemoteProxy[t.Any] | T:
        if instance is None:
            return self
        if self._resolved is None:
            self._resolved = self.get()
            _dependencies.append(self)
        return self._resolved

    def __getattr__(self, name: str) -> t.Any:
        raise AttributeError("Dependency must be accessed as a class attribute")

    async def close(self) -> None:
        if self._resolved is None:
            return

        remote_proxy = t.cast("RemoteProxy[t.Any]", self._resolved)
        if asyncio.iscoroutinefunction(getattr(remote_proxy, "close", None)):
            await remote_proxy.close()


@t.overload
def depends(
    *,
    url: str | None = ...,
    deployment: str | None = ...,
    cluster: str | None = ...,
) -> Dependency[None]: ...


@t.overload
def depends(
    on: Service[T],
    *,
    url: str | None = ...,
    deployment: str | None = ...,
    cluster: str | None = ...,
) -> Dependency[T]: ...


def depends(
    on: Service[T] | None = None,
    *,
    url: str | None = None,
    deployment: str | None = None,
    cluster: str | None = None,
) -> Dependency[T]:
    """Create a dependency on other service or deployment

    Args:
        on: Service[T] | None: The service to depend on.
        url: str | None: The URL of the service to depend on.
        deployment: str | None: The deployment of the service to depend on.
        cluster: str | None: The cluster of the service to depend on.

    Examples:

    .. code-block:: python

        @bentoml.service
        class MyService:
            # depends on a service
            svc_a = bentoml.depends(SVC_A)
            # depends on a deployment
            svc_b = bentoml.depends(deployment="ci-iris")
            # depends on a remote service with url
            svc_c = bentoml.depends(url="http://192.168.1.1:3000")
            # For the latter two cases, the service can be given to provide more accurate types:
            svc_d = bentoml.depends(url="http://192.168.1.1:3000", on=SVC_D)
    """
    if on is not None and not isinstance(on, Service):
        raise TypeError("depends() expects a class decorated with @bentoml.service()")
    return Dependency(on, url=url, deployment=deployment, cluster=cluster)
