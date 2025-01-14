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
from attrs import define  # type: ignore
from attrs import field  # type: ignore
from simple_di import inject  # type: ignore
from typing_extensions import ParamSpec
from typing_extensions import TypeAlias
from typing_extensions import TypeGuard
from typing_extensions import Unpack

from bentoml import Runner
from bentoml._internal import external_typing as ext
from bentoml._internal.bento.bento import Bento
from bentoml._internal.bento.build_config import BentoEnvSchema
from bentoml._internal.configuration.containers import BentoMLContainer
from bentoml._internal.context import ServiceContext
from bentoml._internal.models import Model as StoredModel
from bentoml._internal.service.openapi.specification import OpenAPISpecification
from bentoml._internal.utils import deprecated
from bentoml._internal.utils import dict_filter_none
from bentoml._internal.utils.circus import Server
from bentoml.exceptions import BentoMLConfigException
from bentoml.exceptions import BentoMLException

from ..images import Image
from ..method import APIMethod
from ..models import BentoModel
from ..models import HuggingFaceModel
from ..models import Model
from .config import ServiceConfig as Config
from .types import DependencyInstance
from .types import ServiceInstance
from .types import T
from .types import provide

# Type variables for generic types
_P = ParamSpec("_P")  # Parameters for service methods

if t.TYPE_CHECKING:
    from typing_extensions import TypeGuard

    from bentoml._internal.service.openapi.specification import OpenAPISpecification

    # Type aliases for Service class fields
    ServiceEnvs = t.List[BentoEnvSchema]
    ServiceImportStr = t.Optional[str]

    def is_service_attributes(obj: t.Any) -> TypeGuard[ServiceAttributes[t.Any]]:
        """Type guard for checking if an object has expected Service attributes."""
        return all(
            hasattr(obj, attr)
            for attr in (
                "__name__",
                "__doc__",
                "__bentoml_mounted_apps__",
                "__bentoml_gradio_apps__",
                "__call__",
            )
        )

    # Runtime attribute types for Service class
    class ServiceAttributes(t.Protocol[T]):
        """Protocol defining expected attributes of Service inner class."""

        __name__: str
        __doc__: t.Optional[str]
        __bentoml_mounted_apps__: t.List[t.Any]
        __bentoml_gradio_apps__: t.List[t.Any]
        __call__: t.Callable[[], T]


if t.TYPE_CHECKING:
    pass

logger = logging.getLogger("bentoml.io")

ServiceDecorator: TypeAlias = t.Callable[[type[T]], "Service[T]"]


def with_config(
    func: t.Callable[..., t.Any],
) -> t.Callable[..., t.Any]:
    def wrapper(self: Service[t.Any], *args: t.Any, **kwargs: t.Any) -> t.Any:
        self.inject_config()
        return func(self, *args, **kwargs)

    return wrapper


def convert_envs(envs: t.List[t.Dict[str, t.Any]]) -> t.List[BentoEnvSchema]:
    return [BentoEnvSchema(**env) for env in envs]


@define(auto_attribs=True, slots=False, frozen=False, eq=False)  # type: ignore[misc]
class Service(t.Generic[T]):
    """A Bentoml service that can be served by BentoML server."""

    config: t.Dict[str, t.Any] = field(
        factory=dict,
        init=True,
        repr=True,
        eq=False,
        metadata={"type": "ServiceConfig"},
    )
    inner: t.Type[T] = field(
        init=True,
        repr=True,
        eq=False,
        metadata={"type": "type[T]"},
    )
    image: t.Optional[Image] = field(
        default=None,
        init=True,
        repr=True,
        eq=False,
        metadata={"type": "ServiceImage"},
    )
    envs: t.List[BentoEnvSchema] = field(
        factory=list,
        converter=convert_envs,
        init=True,
        repr=True,
        eq=False,
        metadata={"type": "List[BentoEnvSchema]"},
    )
    bento: t.Optional[Bento] = field(
        init=False,
        default=None,
        repr=False,
        eq=False,
        metadata={"type": "ServiceBento"},
    )
    models: t.List[t.Union[StoredModel, Model[t.Any]]] = field(
        factory=list,
        init=True,
        repr=True,
        eq=False,
        metadata={"type": "ModelList"},
    )
    apis: t.Dict[str, APIMethod[..., t.Any]] = field(
        factory=dict,
        init=True,
        repr=True,
        eq=False,
        metadata={"type": "APIMethodDict"},
    )
    dependencies: t.Dict[str, DependencyInstance] = field(
        factory=dict,
        init=False,
        repr=False,
        eq=False,
        metadata={"type": "DependencyDict"},
    )
    openapi_service_overrides: t.Dict[str, t.Any] = field(
        factory=dict,
        init=True,
        repr=True,
        eq=False,
        metadata={"type": "ComponentDict"},
    )
    mount_apps: t.List[t.Tuple[ext.ASGIApp, str, str]] = field(
        factory=list,
        init=False,
        repr=False,
        eq=False,
        metadata={"type": "MountAppList"},
    )
    middlewares: t.List[t.Tuple[t.Type[ext.AsgiMiddleware], t.Dict[str, t.Any]]] = (
        field(
            factory=list,
            init=False,
            repr=False,
            eq=False,
            metadata={"type": "MiddlewareList"},
        )
    )
    # service context
    context: ServiceContext = field(
        init=False,
        factory=ServiceContext,
        eq=False,
        metadata={"type": "ServiceContext"},
    )
    working_dir: str = field(
        init=False,
        factory=os.getcwd,
        eq=False,
        metadata={"type": "str"},
    )
    # import info
    _caller_module: str = field(
        init=False,
        default="",
        eq=False,
        metadata={"type": "str"},
    )
    _import_str: t.Optional[str] = field(
        init=False,
        default=None,
        eq=False,
        metadata={"type": "typing.Optional[str]"},
    )

    def __attrs_post_init__(self) -> None:
        has_task = False
        inner_attrs = t.cast(ServiceAttributes[T], self.inner)
        for attr_name in dir(inner_attrs):
            value = getattr(inner_attrs, attr_name)
            if isinstance(value, DependencyInstance):
                self.dependencies[attr_name] = value
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
                self.apis[attr_name] = t.cast("APIMethod[..., t.Any]", value)

        if has_task:
            traffic = self.config.setdefault("traffic", {})
            traffic["external_queue"] = True
            traffic.setdefault("concurrency", 1)

        # Handle OpenAPI service-level overrides from config
        if "openapi_service_overrides" in self.config:
            self.openapi_service_overrides = self.config["openapi_service_overrides"]

        pre_mount_apps = getattr(inner_attrs, "__bentoml_mounted_apps__", [])
        if pre_mount_apps:
            self.mount_apps.extend(pre_mount_apps)
            delattr(inner_attrs, "__bentoml_mounted_apps__")

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
    def find_dependent_by_path(self, path: str) -> "Service[t.Any]":
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
        return dependent.on

    def find_dependent_by_name(self, name: str) -> "Service[t.Any]":
        """Find a service by name"""
        try:
            return self.all_services()[name]
        except KeyError:
            raise BentoMLException(f"Service {name} not found") from None

    @property
    def url(self) -> str | None:
        """Get the URL of the service, or None if the service is not served"""
        dependency_map = t.cast(
            dict[str, str], BentoMLContainer.remote_runner_mapping.get()
        )
        url = dependency_map.get(self.name)
        return url.replace("tcp://", "http://") if url is not None else None

    @lru_cache(maxsize=1)
    def all_services(self) -> dict[str, "Service[t.Any]"]:
        """Get a map of the service and all recursive dependencies"""
        services: dict[str, "Service[t.Any]"] = {self.name: self}
        for dependency in self.dependencies.values():
            if dependency.on is None:
                continue
            dependents: dict[str, "Service[t.Any]"] = dependency.on.all_services()
            conflict: str | None = next(
                (
                    str(k)
                    for k, v in dependents.items()
                    if k in services and v is not services[k]
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
        inner_attrs = t.cast(ServiceAttributes[T], self.inner)
        gradio_apps = getattr(inner_attrs, "__bentoml_gradio_apps__", [])
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
            delattr(inner_attrs, "__bentoml_gradio_apps__")

    def __call__(self) -> t.Union[T, ServiceInstance]:
        try:
            instance = self.inner()
            # Cast to Any to avoid type checking issues with dynamic attribute assignment
            instance_any = t.cast(t.Any, instance)
            api_keys = list(self.apis.keys())
            instance_any.to_async = _AsyncWrapper(instance, api_keys)
            instance_any.to_sync = _SyncWrapper(instance, api_keys)
            return t.cast(t.Union[T, ServiceInstance], instance)
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
        port: int = provide[BentoMLContainer.http.port],
        host: str = provide[BentoMLContainer.http.host],
        backlog: int = provide[BentoMLContainer.api_server_config.backlog],
        timeout: int | None = None,
        ssl_certfile: str | None = provide[BentoMLContainer.ssl.certfile],
        ssl_keyfile: str | None = provide[BentoMLContainer.ssl.keyfile],
        ssl_keyfile_password: str | None = provide[
            BentoMLContainer.ssl.keyfile_password
        ],
        ssl_version: int | None = provide[BentoMLContainer.ssl.version],
        ssl_cert_reqs: int | None = provide[BentoMLContainer.ssl.cert_reqs],
        ssl_ca_certs: str | None = provide[BentoMLContainer.ssl.ca_certs],
        ssl_ciphers: str | None = provide[BentoMLContainer.ssl.ciphers],
        bentoml_home: str = provide[BentoMLContainer.bentoml_home],
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
    inner: None = None,
    /,
    *,
    image: Image | None = None,
    envs: list[dict[str, t.Any]] | None = None,
    **kwargs: Unpack[Config],
) -> ServiceDecorator[T]: ...


def service(
    inner: type[T] | None = None,
    /,
    *,
    image: Image | None = None,
    envs: list[dict[str, t.Any]] | None = None,
    **kwargs: Unpack[Config],
) -> t.Union[Service[T], ServiceDecorator[T]]:
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

            async def wrapped(*args: _P.args, **kwargs: _P.kwargs) -> t.Any:
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

            def wrapped(*args: _P.args, **kwargs: _P.kwargs) -> t.Any:
                loop = asyncio.get_event_loop()
                return loop.run_until_complete(func(*args, **kwargs))

            return wrapped
