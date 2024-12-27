from __future__ import annotations

import asyncio
import inspect
import logging
import math
import os
import pathlib
import sys
import typing as t
from functools import lru_cache
from functools import partial

import anyio.to_thread
import attrs
from simple_di import Provide
from simple_di import inject
from typing_extensions import Unpack

from bentoml import Runner
from bentoml._internal.bento.bento import Bento
from bentoml._internal.bento.build_config import BentoEnvSchema
from bentoml._internal.configuration.containers import BentoMLContainer
from bentoml._internal.context import ServiceContext
from bentoml._internal.models import Model as StoredModel
from bentoml._internal.utils import deprecated
from bentoml._internal.utils import dict_filter_none
from bentoml.exceptions import BentoMLConfigException
from bentoml.exceptions import BentoMLException

from ..images import Image
from ..method import APIMethod
from ..models import BentoModel
from ..models import HuggingFaceModel
from ..models import Model
from .config import ServiceConfig as Config

logger = logging.getLogger("bentoml.io")

T = t.TypeVar("T", bound=object)

if t.TYPE_CHECKING:
    from bentoml._internal import external_typing as ext
    from bentoml._internal.service.openapi.specification import OpenAPISpecification
    from bentoml._internal.utils.circus import Server

    from .dependency import Dependency

    P = t.ParamSpec("P")
    R = t.TypeVar("R")

    class _ServiceDecorator(t.Protocol):
        def __call__(self, inner: type[T]) -> Service[T]: ...


def with_config(
    func: t.Callable[t.Concatenate["Service[t.Any]", P], R],
) -> t.Callable[t.Concatenate["Service[t.Any]", P], R]:
    def wrapper(self: Service[t.Any], *args: P.args, **kwargs: P.kwargs) -> R:
        self.inject_config()
        return func(self, *args, **kwargs)

    return wrapper


def convert_envs(envs: t.List[t.Dict[str, t.Any]]) -> t.List[BentoEnvSchema]:
    return [BentoEnvSchema(**env) for env in envs]


@attrs.define
class Service(t.Generic[T]):
    """A Bentoml service that can be served by BentoML server."""

    config: Config
    inner: type[T]
    image: t.Optional[Image] = None
    envs: t.List[BentoEnvSchema] = attrs.field(factory=list, converter=convert_envs)
    bento: t.Optional[Bento] = attrs.field(init=False, default=None)
    models: list[Model[t.Any]] = attrs.field(factory=list)
    apis: dict[str, APIMethod[..., t.Any]] = attrs.field(factory=dict)
    dependencies: dict[str, Dependency[t.Any]] = attrs.field(factory=dict, init=False)
    mount_apps: list[tuple[ext.ASGIApp, str, str]] = attrs.field(
        factory=list, init=False
    )
    middlewares: list[tuple[type[ext.AsgiMiddleware], dict[str, t.Any]]] = attrs.field(
        factory=list, init=False
    )
    # service context
    context: ServiceContext = attrs.field(init=False, factory=ServiceContext)
    working_dir: str = attrs.field(init=False, factory=os.getcwd)
    # import info
    _caller_module: str = attrs.field(init=False)
    _import_str: str | None = attrs.field(init=False, default=None)

    def __attrs_post_init__(self) -> None:
        from .dependency import Dependency

        has_task = False
        for field in dir(self.inner):
            value = getattr(self.inner, field)
            if isinstance(value, Dependency):
                self.dependencies[field] = value
            elif isinstance(value, StoredModel):
                logger.warning(
                    "`bentoml.models.get()` as the class attribute is not recommended because it requires the model"
                    f" to exist at import time. Use `{value._attr} = BentoModel({str(value.tag)!r})` instead."
                )
                self.models.append(BentoModel(value.tag))
            elif isinstance(value, Model):
                self.models.append(t.cast(Model[t.Any], value))
            elif isinstance(value, APIMethod):
                if value.is_task:
                    has_task = True
                self.apis[field] = t.cast("APIMethod[..., t.Any]", value)

        if has_task:
            traffic = self.config.setdefault("traffic", {})
            traffic["external_queue"] = True
            traffic.setdefault("concurrency", 1)

        pre_mount_apps = getattr(self.inner, "__bentoml_mounted_apps__", [])
        if pre_mount_apps:
            self.mount_apps.extend(pre_mount_apps)
            delattr(self.inner, "__bentoml_mounted_apps__")

    def __hash__(self):
        return hash(self.name)

    @_caller_module.default  # type: ignore
    def _get_caller_module(self) -> str:
        if __name__ == "__main__":
            return __name__
        current_frame = inspect.currentframe()
        frame = current_frame
        while frame:
            this_name = frame.f_globals["__name__"]
            if this_name != __name__:
                return this_name
            frame = frame.f_back
        return __name__

    def __repr__(self) -> str:
        return f"<{self.__class__.__name__} name={self.name!r}>"

    @lru_cache
    def find_dependent_by_path(self, path: str) -> Service[t.Any]:
        """Find a service by path"""
        attr_name, _, path = path.partition(".")
        if attr_name not in self.dependencies:
            if attr_name in self.all_services():
                return self.all_services()[attr_name]
            else:
                raise BentoMLException(f"Service {attr_name} not found")
        dependent = self.dependencies[attr_name]
        if dependent.on is None:
            raise BentoMLException(f"Service {attr_name} not found")
        if path:
            return dependent.on.find_dependent_by_path(path)
        return dependent

    def find_dependent_by_name(self, name: str) -> Service[t.Any]:
        """Find a service by name"""
        try:
            return self.all_services()[name]
        except KeyError:
            raise BentoMLException(f"Service {name} not found") from None

    @property
    def url(self) -> str | None:
        """Get the URL of the service, or None if the service is not served"""
        dependency_map = BentoMLContainer.remote_runner_mapping.get()
        url = dependency_map.get(self.name)
        return url.replace("tcp://", "http://") if url else None

    @lru_cache(maxsize=1)
    def all_services(self) -> dict[str, Service[t.Any]]:
        """Get a map of the service and all recursive dependencies"""
        services: dict[str, Service[t.Any]] = {self.name: self}
        for dependency in self.dependencies.values():
            if dependency.on is None:
                continue
            dependents = dependency.on.all_services()
            conflict = next(
                (
                    k
                    for k in dependents
                    if k in services and dependents[k] is not services[k]
                ),
                None,
            )
            if conflict:
                raise BentoMLConfigException(
                    f"Dependency conflict: {conflict} is already defined by {services[conflict].inner}"
                )
            services.update(dependents)
        return services

    @property
    def doc(self) -> str:
        from bentoml._internal.bento.bento import get_default_svc_readme

        if self.bento is not None:
            return self.bento.doc

        return get_default_svc_readme(self)

    def schema(self) -> dict[str, t.Any]:
        return dict_filter_none(
            {
                "name": self.name,
                "type": "service",
                "routes": [method.schema() for method in self.apis.values()],
                "description": getattr(self.inner, "__doc__", None),
            }
        )

    @property
    def name(self) -> str:
        name = self.config.get("name") or self.inner.__name__
        return name

    @property
    def import_string(self) -> str:
        if self._import_str is None:
            import_module = self._caller_module
            if import_module == "__main__":
                if hasattr(sys.modules["__main__"], "__file__"):
                    import_module = sys.modules["__main__"].__file__
                    assert isinstance(import_module, str)
                    try:
                        import_module_path = pathlib.Path(import_module).relative_to(
                            self.working_dir
                        )
                    except ValueError:
                        raise BentoMLException(
                            "Failed to get service import origin, service object defined in __main__ module is not supported"
                        )
                    import_module = str(import_module_path.with_suffix("")).replace(
                        os.path.sep, "."
                    )
                else:
                    raise BentoMLException(
                        "Failed to get service import origin, service object defined interactively in console or notebook is not supported"
                    )

            if self._caller_module not in sys.modules:
                raise BentoMLException(
                    "Failed to get service import origin, service object must be defined in a module"
                )

            for name, value in vars(sys.modules[self._caller_module]).items():
                if value is self:
                    self._import_str = f"{import_module}:{name}"
                    break
            else:
                raise BentoMLException(
                    "Failed to get service import origin, service object must be assigned to a variable at module level"
                )
        return self._import_str

    def to_asgi(self, is_main: bool = True) -> ext.ASGIApp:
        from _bentoml_impl.server.app import ServiceAppFactory

        self.inject_config()
        factory = ServiceAppFactory(self, is_main=is_main)
        return factory()

    def mount_asgi_app(
        self, app: ext.ASGIApp, path: str = "/", name: str | None = None
    ) -> None:
        self.mount_apps.append((app, path, name))  # type: ignore

    def mount_wsgi_app(
        self, app: ext.WSGIApp, path: str = "/", name: str | None = None
    ) -> None:
        from a2wsgi import WSGIMiddleware

        self.mount_apps.append((WSGIMiddleware(app), path, name))  # type: ignore

    def add_asgi_middleware(
        self, middleware_cls: type[ext.AsgiMiddleware], **options: t.Any
    ) -> None:
        self.middlewares.append((middleware_cls, options))

    def gradio_app_startup_hook(self, max_concurrency: int):
        gradio_apps = getattr(self.inner, "__bentoml_gradio_apps__", [])
        if gradio_apps:
            for gradio_app, path, _ in gradio_apps:
                logger.info(f"Initializing gradio app at: {path or '/'}")
                blocks = gradio_app.get_blocks()
                blocks.queue(default_concurrency_limit=max_concurrency)
                if hasattr(blocks, "startup_events"):
                    # gradio < 5.0
                    blocks.startup_events()
                else:
                    # gradio >= 5.0
                    blocks.run_startup_events()
            delattr(self.inner, "__bentoml_gradio_apps__")

    def __call__(self) -> T:
        try:
            instance = self.inner()
            instance.to_async = _AsyncWrapper(instance, self.apis.keys())
            instance.to_sync = _SyncWrapper(instance, self.apis.keys())
            return instance
        except Exception:
            logger.exception("Initializing service error")
            raise

    @property
    def openapi_spec(self) -> OpenAPISpecification:
        from .openapi import generate_spec

        return generate_spec(self)

    def inject_config(self) -> None:
        from bentoml._internal.configuration import load_config
        from bentoml._internal.configuration.containers import BentoMLContainer
        from bentoml._internal.configuration.containers import config_merger

        # XXX: ensure at least one item to make `flatten_dict` work
        override_defaults = {
            "services": {
                name: (svc.config or {"workers": 1})
                for name, svc in self.all_services().items()
            }
        }

        load_config(override_defaults=override_defaults, use_version=2)
        main_config = BentoMLContainer.config.services[self.name].get()
        api_server_keys = (
            "traffic",
            "metrics",
            "logging",
            "ssl",
            "http",
            "grpc",
            "backlog",
            "runner_probe",
            "max_runner_connections",
        )
        api_server_config = {
            k: main_config[k] for k in api_server_keys if main_config.get(k) is not None
        }
        rest_config = {
            k: main_config[k] for k in main_config if k not in api_server_keys
        }
        existing = t.cast(t.Dict[str, t.Any], BentoMLContainer.config.get())
        config_merger.merge(existing, {"api_server": api_server_config, **rest_config})
        BentoMLContainer.config.set(existing)  # type: ignore

    @with_config
    @inject
    def serve_http(
        self,
        *,
        working_dir: str | None = None,
        port: int = Provide[BentoMLContainer.http.port],
        host: str = Provide[BentoMLContainer.http.host],
        backlog: int = Provide[BentoMLContainer.api_server_config.backlog],
        timeout: int | None = None,
        ssl_certfile: str | None = Provide[BentoMLContainer.ssl.certfile],
        ssl_keyfile: str | None = Provide[BentoMLContainer.ssl.keyfile],
        ssl_keyfile_password: str | None = Provide[
            BentoMLContainer.ssl.keyfile_password
        ],
        ssl_version: int | None = Provide[BentoMLContainer.ssl.version],
        ssl_cert_reqs: int | None = Provide[BentoMLContainer.ssl.cert_reqs],
        ssl_ca_certs: str | None = Provide[BentoMLContainer.ssl.ca_certs],
        ssl_ciphers: str | None = Provide[BentoMLContainer.ssl.ciphers],
        bentoml_home: str = Provide[BentoMLContainer.bentoml_home],
        development_mode: bool = False,
        reload: bool = False,
        threaded: bool = False,
    ) -> Server:
        from _bentoml_impl.server import serve_http
        from bentoml._internal.log import configure_logging

        configure_logging()

        return serve_http(
            self,
            working_dir=working_dir,
            host=host,
            port=port,
            backlog=backlog,
            timeout=timeout,
            ssl_certfile=ssl_certfile,
            ssl_keyfile=ssl_keyfile,
            ssl_keyfile_password=ssl_keyfile_password,
            ssl_version=ssl_version,
            ssl_cert_reqs=ssl_cert_reqs,
            ssl_ca_certs=ssl_ca_certs,
            ssl_ciphers=ssl_ciphers,
            bentoml_home=bentoml_home,
            development_mode=development_mode,
            reload=reload,
            threaded=threaded,
        )

    def on_load_bento(self, bento: Bento) -> None:
        service_info = next(svc for svc in bento.info.services if svc.name == self.name)
        for model, info in zip(self.models, service_info.models):
            # Replace the model version with the one in the Bento
            if not isinstance(model, HuggingFaceModel):
                continue
            model_id = info.metadata.get("model_id")  # use the case in bento info
            if not model_id:
                model_id = info.tag.name.replace("--", "/")
            revision = info.metadata.get("revision", info.tag.version)
            model.model_id = model_id
            model.revision = revision
        self.bento = bento


@t.overload
def service(inner: type[T], /) -> Service[T]: ...


@t.overload
def service(
    inner: None = ...,
    /,
    *,
    image: Image | None = None,
    envs: list[dict[str, t.Any]] | None = None,
    **kwargs: Unpack[Config],
) -> _ServiceDecorator: ...


def service(
    inner: type[T] | None = None,
    /,
    *,
    image: Image | None = None,
    envs: list[dict[str, t.Any]] | None = None,
    **kwargs: Unpack[Config],
) -> t.Any:
    """Mark a class as a BentoML service.

    Example:

        @service(traffic={"timeout": 60})
        class InferenceService:
            @api
            def predict(self, input: str) -> str:
                return input
    """
    config = kwargs

    def decorator(inner: type[T]) -> Service[T]:
        if isinstance(inner, Service):
            raise TypeError("service() decorator can only be applied once")
        return Service(config=config, inner=inner, image=image, envs=envs or [])

    return decorator(inner) if inner is not None else decorator


@deprecated()
def runner_service(runner: Runner, **kwargs: Unpack[Config]) -> Service[t.Any]:
    """Make a service from a legacy Runner"""
    if not isinstance(runner, Runner):  # type: ignore
        raise ValueError(f"Expect an instance of Runner, but got {type(runner)}")

    class RunnerHandle(runner.runnable_class):
        def __init__(self) -> None:
            super().__init__(**runner.runnable_init_params)

    RunnerHandle.__name__ = runner.name
    apis: dict[str, APIMethod[..., t.Any]] = {}
    assert runner.runnable_class.bentoml_runnable_methods__ is not None
    for method in runner.runner_methods:
        runnable_method = runner.runnable_class.bentoml_runnable_methods__[method.name]
        api = APIMethod(  # type: ignore
            func=runnable_method.func,
            batchable=runnable_method.config.batchable,
            batch_dim=runnable_method.config.batch_dim,
            max_batch_size=method.max_batch_size,
            max_latency_ms=method.max_latency_ms,
        )
        apis[method.name] = api
    config: Config = {}
    resource_config = runner.resource_config or {}
    if (
        "nvidia.com/gpu" in runner.runnable_class.SUPPORTED_RESOURCES
        and "nvidia.com/gpu" in resource_config
    ):
        gpus: list[int] | str | int = resource_config["nvidia.com/gpu"]
        if isinstance(gpus, str):
            gpus = int(gpus)
        if runner.workers_per_resource > 1:
            config["workers"] = {}
            workers_per_resource = int(runner.workers_per_resource)
            if isinstance(gpus, int):
                gpus = list(range(gpus))
            for i in gpus:
                config["workers"].extend([{"gpus": i}] * workers_per_resource)
        else:
            resources_per_worker = int(1 / runner.workers_per_resource)
            if isinstance(gpus, int):
                config["workers"] = [
                    {"gpus": resources_per_worker}
                    for _ in range(gpus // resources_per_worker)
                ]
            else:
                config["workers"] = [
                    {"gpus": gpus[i : i + resources_per_worker]}
                    for i in range(0, len(gpus), resources_per_worker)
                ]
    elif "cpus" in resource_config:
        config["workers"] = (
            math.ceil(resource_config["cpus"]) * runner.workers_per_resource
        )
    config.update(kwargs)
    return Service(
        config=config,
        inner=RunnerHandle,
        models=[BentoModel(m.tag) for m in runner.models],
        apis=apis,
    )


class _Wrapper:
    def __init__(self, wrapped: t.Any, apis: t.Iterable[str]) -> None:
        self.__call = None
        for name in apis:
            if name == "__call__":
                self.__call = self._make_method(wrapped, name)
            else:
                setattr(self, name, self._make_method(wrapped, name))

    def __call__(self, *args: t.Any, **kwargs: t.Any) -> t.Any:
        if self.__call is None:
            raise TypeError("This service is not callable.")
        return self.__call(*args, **kwargs)

    def _make_method(self, instance: t.Any, name: str) -> t.Any:
        raise NotImplementedError


class _AsyncWrapper(_Wrapper):
    def _make_method(self, instance: t.Any, name: str) -> t.Any:
        original_func = func = getattr(instance, name).local
        while hasattr(original_func, "func"):
            original_func = original_func.func
        is_async_func = (
            asyncio.iscoroutinefunction(original_func)
            or (
                callable(original_func)
                and asyncio.iscoroutinefunction(original_func.__call__)  # type: ignore
            )
            or inspect.isasyncgenfunction(original_func)
        )
        if is_async_func:
            return func

        if inspect.isgeneratorfunction(original_func):

            async def wrapped_gen(
                *args: t.Any, **kwargs: t.Any
            ) -> t.AsyncGenerator[t.Any, None]:
                gen = func(*args, **kwargs)
                next_fun = gen.__next__
                while True:
                    try:
                        yield await anyio.to_thread.run_sync(next_fun)
                    except StopIteration:
                        break
                    except RuntimeError as e:
                        if "raised StopIteration" in str(e):
                            break
                        raise

            return wrapped_gen
        else:

            async def wrapped(*args: P.args, **kwargs: P.kwargs) -> t.Any:
                return await anyio.to_thread.run_sync(partial(func, **kwargs), *args)

            return wrapped


class _SyncWrapper(_Wrapper):
    def _make_method(self, instance: t.Any, name: str) -> t.Any:
        original_func = func = getattr(instance, name).local
        while hasattr(original_func, "func"):
            original_func = original_func.func
        is_async_func = (
            asyncio.iscoroutinefunction(original_func)
            or (
                callable(original_func)
                and asyncio.iscoroutinefunction(original_func.__call__)  # type: ignore
            )
            or inspect.isasyncgenfunction(original_func)
        )
        if not is_async_func:
            return func

        if inspect.isasyncgenfunction(original_func):

            def wrapped_gen(
                *args: t.Any, **kwargs: t.Any
            ) -> t.Generator[t.Any, None, None]:
                agen = func(*args, **kwargs)
                loop = asyncio.get_event_loop()
                while True:
                    try:
                        yield loop.run_until_complete(agen.__anext__())
                    except StopAsyncIteration:
                        break

            return wrapped_gen
        else:

            def wrapped(*args: P.args, **kwargs: P.kwargs) -> t.Any:
                loop = asyncio.get_event_loop()
                return loop.run_until_complete(func(*args, **kwargs))

            return wrapped
