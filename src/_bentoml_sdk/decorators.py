from __future__ import annotations

import typing as t

from bentoml._internal.types import LazyType

from .io_models import IODescriptor
from .method import APIMethod

F = t.TypeVar("F", bound=t.Callable[..., t.Any])
R = t.TypeVar("R")
T = t.TypeVar("T", bound="APIMethod[..., t.Any]")

if t.TYPE_CHECKING:
    from fastapi import FastAPI  # noqa: F401

    from bentoml._internal.external_typing import ASGIApp

    P = t.ParamSpec("P")
else:
    P = t.TypeVar("P")


def on_shutdown(func: F) -> F:
    """Mark a method as a shutdown hook for the service."""
    setattr(func, "__bentoml_shutdown_hook__", True)
    return func


def on_startup(func: F) -> F:
    """Mark a method as a startup hook for the service."""
    setattr(func, "__bentoml_startup_hook__", True)
    return func


def on_deployment(func: t.Callable[P, R] | staticmethod[P, R]) -> staticmethod[P, R]:
    inner = func.__func__ if isinstance(func, staticmethod) else func
    setattr(inner, "__bentoml_deployment_hook__", True)
    return func if isinstance(func, staticmethod) else staticmethod(func)  # type: ignore


@t.overload
def api(func: t.Callable[t.Concatenate[t.Any, P], R]) -> APIMethod[P, R]: ...


@t.overload
def api(
    *,
    route: str | None = ...,
    name: str | None = ...,
    input_spec: type[IODescriptor] | None = ...,
    output_spec: type[IODescriptor] | None = ...,
    batchable: bool = ...,
    batch_dim: int | tuple[int, int] = ...,
    max_batch_size: int = ...,
    max_latency_ms: int = ...,
) -> t.Callable[[t.Callable[t.Concatenate[t.Any, P], R]], APIMethod[P, R]]: ...


def api(
    func: t.Callable[t.Concatenate[t.Any, P], R] | None = None,
    *,
    route: str | None = None,
    name: str | None = None,
    input_spec: type[IODescriptor] | None = None,
    output_spec: type[IODescriptor] | None = None,
    batchable: bool = False,
    batch_dim: int | tuple[int, int] = 0,
    max_batch_size: int = 100,
    max_latency_ms: int = 60000,
) -> (
    APIMethod[P, R]
    | t.Callable[[t.Callable[t.Concatenate[t.Any, P], R]], APIMethod[P, R]]
):
    """Make a BentoML API method.
    This decorator can be used either with or without arguments.

    Args:
        func: The function to be wrapped.
        route: The route of the API. e.g. "/predict"
        name: The name of the API.
        input_spec: The input spec of the API, should be a subclass of ``pydantic.BaseModel``.
        output_spec: The output spec of the API, should be a subclass of ``pydantic.BaseModel``.
        batchable: Whether the API is batchable.
        batch_dim: The batch dimension of the API.
        max_batch_size: The maximum batch size of the API.
        max_latency_ms: The maximum latency of the API.
    """

    def wrapper(func: t.Callable[t.Concatenate[t.Any, P], R]) -> APIMethod[P, R]:
        if func.__name__.startswith("__"):
            raise ValueError("API methods cannot start with '__'")
        params: dict[str, t.Any] = {
            "batchable": batchable,
            "batch_dim": batch_dim,
            "max_batch_size": max_batch_size,
            "max_latency_ms": max_latency_ms,
        }
        if route is not None:
            params["route"] = route
        if input_spec is not None:
            params["input_spec"] = input_spec
        if output_spec is not None:
            params["output_spec"] = output_spec
        return APIMethod(func, **params)

    if func is not None:
        return wrapper(func)
    return wrapper


def asgi_app(
    app: ASGIApp, *, path: str = "/", name: str | None = None
) -> t.Callable[[R], R]:
    """Mount an ASGI app to the service.

    Args:
        app: The ASGI app to be mounted.
        path: The path to mount the app.
        name: The name of the app.
    """

    from ._internals import make_fastapi_class_views
    from .service import Service

    def decorator(obj: R) -> R:
        lazy_fastapi = LazyType["FastAPI"]("fastapi.FastAPI")

        if isinstance(obj, Service):
            obj.mount_asgi_app(app, path=path, name=name)
            if lazy_fastapi.isinstance(app):
                make_fastapi_class_views(obj.inner, app)
        else:
            mount_apps = getattr(obj, "__bentoml_mounted_apps__", [])
            mount_apps.append((app, path, name))
            setattr(obj, "__bentoml_mounted_apps__", mount_apps)
            if lazy_fastapi.isinstance(app):
                make_fastapi_class_views(obj, app)
        return obj

    return decorator


@t.overload
def task(func: t.Callable[t.Concatenate[t.Any, P], R]) -> APIMethod[P, R]: ...


@t.overload
def task(
    *,
    route: str | None = ...,
    name: str | None = ...,
    input_spec: type[IODescriptor] | None = ...,
    output_spec: type[IODescriptor] | None = ...,
    batchable: bool = ...,
    batch_dim: int | tuple[int, int] = ...,
    max_batch_size: int = ...,
    max_latency_ms: int = ...,
) -> t.Callable[[t.Callable[t.Concatenate[t.Any, P], R]], APIMethod[P, R]]: ...


def task(
    func: t.Callable[t.Concatenate[t.Any, P], R] | None = None,
    *,
    route: str | None = None,
    name: str | None = None,
    input_spec: type[IODescriptor] | None = None,
    output_spec: type[IODescriptor] | None = None,
    batchable: bool = False,
    batch_dim: int | tuple[int, int] = 0,
    max_batch_size: int = 100,
    max_latency_ms: int = 60000,
) -> (
    APIMethod[P, R]
    | t.Callable[[t.Callable[t.Concatenate[t.Any, P], R]], APIMethod[P, R]]
):
    """Mark a method as a BentoML async task.
    This decorator can be used either with or without arguments.

    Args:
        func: The function to be wrapped.
        route: The route of the API. e.g. "/predict"
        name: The name of the API.
        input_spec: The input spec of the API, should be a subclass of ``pydantic.BaseModel``.
        output_spec: The output spec of the API, should be a subclass of ``pydantic.BaseModel``.
        batchable: Whether the API is batchable.
        batch_dim: The batch dimension of the API.
        max_batch_size: The maximum batch size of the API.
        max_latency_ms: The maximum latency of the API.
    """

    def wrapper(func: t.Callable[t.Concatenate[t.Any, P], R]) -> APIMethod[P, R]:
        if func.__name__.startswith("__"):
            raise ValueError("API methods cannot start with '__'")
        params: dict[str, t.Any] = {
            "batchable": batchable,
            "batch_dim": batch_dim,
            "max_batch_size": max_batch_size,
            "max_latency_ms": max_latency_ms,
        }
        if route is not None:
            params["route"] = route
        if input_spec is not None:
            params["input_spec"] = input_spec
        if output_spec is not None:
            params["output_spec"] = output_spec
        meth = APIMethod(func, **params, is_task=True)
        if meth.is_stream:
            raise ValueError("Async task cannot return a stream.")
        return meth

    if func is not None:
        return wrapper(func)
    return wrapper
