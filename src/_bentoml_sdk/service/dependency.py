"""Dependency management for BentoML services."""

from __future__ import annotations

import typing as t

if t.TYPE_CHECKING:
    from .factory import Service

from .types import Dependency


@t.overload
def depends(
    *,
    url: str | None = ...,
    deployment: str | None = ...,
    cluster: str | None = ...,
) -> Dependency[None]: ...


@t.overload
def depends(
    on: t.Any,
    *,
    url: str | None = ...,
    deployment: str | None = ...,
    cluster: str | None = ...,
) -> Dependency[t.Any]: ...


def depends(
    on: t.Optional["Service[t.Any]"] = None,  # Forward reference to Service
    *,
    url: str | None = None,
    deployment: str | None = None,
    cluster: str | None = None,
) -> "Dependency[t.Any]":
    """Create a dependency on other service or deployment.

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
    if on is not None and not hasattr(on, "__bentoml_service__"):
        raise TypeError("depends() expects a class decorated with @bentoml.service()")
    return Dependency(
        on=on,
        url=url,
        deployment=deployment,
        cluster=cluster,
    )
