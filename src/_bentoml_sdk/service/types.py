"""Type definitions for BentoML service dependencies."""

from __future__ import annotations

import asyncio
import typing as t
from typing import Any
from typing import TypeVar
from typing import cast
from typing import runtime_checkable

from simple_di import Provide  # type: ignore
from typing_extensions import Protocol

from bentoml._internal.bento.build_config import BentoEnvSchema
from bentoml._internal.configuration.containers import BentoMLContainer
from bentoml._internal.context import ServiceContext as _ServiceContext
from bentoml._internal.external_typing import AsgiMiddleware
from bentoml.exceptions import BentoMLException

if t.TYPE_CHECKING:
    from starlette.types import ASGIApp

# Type variables and aliases
T = TypeVar("T")  # Service type parameter

# Basic type aliases
StrNone = t.Optional[str]
ServiceStr = str
ServiceImportStr = t.Optional[str]

# For use in parameter defaults
provide = cast(Any, Provide)


@runtime_checkable
class DependencyInstance(Protocol):
    """Protocol for dependency instances."""

    to_async: Any
    to_sync: Any
    on: t.Optional["Service[t.Any]"]  # Forward reference to Service


@runtime_checkable
class ServiceInstance(Protocol):
    """Protocol for service instances."""

    to_async: Any
    to_sync: Any


# Forward references for complex types
if t.TYPE_CHECKING:
    from _bentoml_impl.client.proxy import RemoteProxy
    from bentoml._internal.bento.bento import Bento
    from bentoml._internal.models import Model as StoredModel

    from ..images import Image
    from ..method import APIMethod
    from ..models import Model

# Forward reference for Service type to avoid circular imports
if t.TYPE_CHECKING:
    from .factory import Service

# Type aliases that work both at runtime and type checking
ServiceNone = t.Optional["Service[t.Any]"]
ServiceConfig = t.Dict[str, t.Any]
APIMethodDict = t.Dict[str, "APIMethod[..., t.Any]"]
DependencyDict = t.Dict[str, "DependencyInstance"]
ComponentDict = t.Dict[str, t.Any]
ModelList = t.List[t.Union["StoredModel", "Model[t.Any]"]]
MountAppList = t.List[t.Tuple["ASGIApp", str, str]]
MiddlewareList = t.List[t.Tuple[t.Type["AsgiMiddleware"], t.Dict[str, t.Any]]]
EnvList = t.List[BentoEnvSchema]
ServiceImage = t.Optional["Image"]
ServiceBento = t.Optional["Bento"]
ServiceContextType = _ServiceContext


# Protocol definitions already defined above


# Global list of dependencies for cleanup
_dependencies: list["Dependency[t.Any]"] = []


async def cleanup() -> None:
    """Clean up all dependencies."""
    tasks = [dep.close() for dep in _dependencies]
    await asyncio.gather(*tasks)
    _dependencies.clear()


# No need for attrs decorator since we're using manual __init__
class Dependency(t.Generic[T]):
    """A dependency on another service or deployment.

    This class represents a dependency on another BentoML service or deployment.
    The dependency can be resolved at runtime to either a local service instance
    or a remote proxy to the service.
    """

    if t.TYPE_CHECKING:
        from .factory import Service

    on: t.Optional["Service[t.Any]"] = None  # Forward reference to Service
    url: t.Optional[str] = None
    deployment: t.Optional[str] = None
    cluster: t.Optional[str] = None
    _resolved: t.Any = None

    def __init__(
        self,
        *,
        on: t.Optional["Service[t.Any]"] = None,  # Forward reference to Service
        url: t.Optional[str] = None,
        deployment: t.Optional[str] = None,
        cluster: t.Optional[str] = None,
    ) -> None:
        """Initialize a dependency.

        Args:
            on: The service to depend on
            url: The URL of the service
            deployment: The deployment name
            cluster: The cluster name
        """
        self.on = on
        self.url = url
        self.deployment = deployment
        self.cluster = cluster
        self._resolved = None

    @t.overload
    def get(
        self: "Dependency[None]",
        *,
        runner_mapping: dict[str, str] = ...,
    ) -> "RemoteProxy[t.Any]": ...

    @t.overload
    def get(
        self: "Dependency[T]",
        *,
        runner_mapping: dict[str, str] = ...,
    ) -> T: ...

    def get(
        self: "Dependency[None] | Dependency[T]",
        *,
        runner_mapping: dict[str, str] = provide[
            BentoMLContainer.remote_runner_mapping
        ],
    ) -> t.Union[T, "RemoteProxy[t.Any]"]:
        """Get the dependency instance."""
        from _bentoml_impl.client.proxy import RemoteProxy

        media_type = "application/json"
        if self.deployment and self.url:
            raise BentoMLException("Cannot specify both deployment and url")
        if self.deployment:
            client = BentoMLContainer.rest_api_client.get()  # type: ignore
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
                instance: t.Any = self.on()
                if isinstance(instance, DependencyInstance):
                    return t.cast(T, instance)
                return t.cast(T, instance)

        if self.url is None:
            raise BentoMLException("URL must be set before creating RemoteProxy")
        proxy: RemoteProxy[t.Any] = RemoteProxy(
            str(self.url), service=self.on, media_type=media_type
        )
        return proxy.as_service()

    @t.overload
    def __get__(
        self: "Dependency[T]", instance: None, owner: t.Any
    ) -> "Dependency[T]": ...

    @t.overload
    def __get__(self: "Dependency[T]", instance: t.Any, owner: t.Any) -> T: ...

    def __get__(
        self: "Dependency[T]", instance: t.Any, owner: t.Any
    ) -> t.Union["Dependency[T]", T]:
        """Descriptor protocol implementation."""
        if instance is None:
            return self
        if self._resolved is None:
            self._resolved = self.get()
            _dependencies.append(self)
        return t.cast(T, self._resolved)

    def __getattr__(self, name: str) -> t.Any:
        raise AttributeError("Dependency must be accessed as a class attribute")

    async def close(self) -> None:
        """Close the dependency."""
        if self._resolved is None:
            return

        remote_proxy = t.cast(RemoteProxy[t.Any], self._resolved)
        if asyncio.iscoroutinefunction(getattr(remote_proxy, "close", None)):
            await remote_proxy.close()
