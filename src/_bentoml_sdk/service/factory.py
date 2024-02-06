from __future__ import annotations

import inspect
import logging
import math
import os
import pathlib
import sys
import typing as t
from functools import lru_cache
from functools import partial

import attrs
from simple_di import Provide
from simple_di import inject
from typing_extensions import Unpack

from bentoml import Runner
from bentoml._internal.bento.bento import Bento
from bentoml._internal.configuration.containers import BentoMLContainer
from bentoml._internal.context import ServiceContext
from bentoml._internal.models import Model
from bentoml._internal.utils import dict_filter_none
from bentoml.exceptions import BentoMLException

from ..api import APIMethod
from .config import ServiceConfig as Config
from .config import validate

logger = logging.getLogger("bentoml.io")

T = t.TypeVar("T", bound=object)

if t.TYPE_CHECKING:
    from bentoml._internal import external_typing as ext
    from bentoml._internal.service.openapi.specification import OpenAPISpecification
    from bentoml._internal.types import LifecycleHook

    from .dependency import Dependency

    P = t.ParamSpec("P")
    R = t.TypeVar("R")

    ContextFunc = t.Callable[[ServiceContext], None | t.Coroutine[t.Any, t.Any, None]]
    HookF = t.TypeVar("HookF", bound=LifecycleHook)
    HookF_ctx = t.TypeVar("HookF_ctx", bound=ContextFunc)

    class _ServiceDecorator(t.Protocol):
        def __call__(self, inner: type[T]) -> Service[T]:
            ...


def with_config(
    func: t.Callable[t.Concatenate["Service[t.Any]", P], R],
) -> t.Callable[t.Concatenate["Service[t.Any]", P], R]:
    def wrapper(self: Service[t.Any], *args: P.args, **kwargs: P.kwargs) -> R:
        self.inject_config()
        return func(self, *args, **kwargs)

    return wrapper


@attrs.define
class Service(t.Generic[T]):
    """A Bentoml service that can be served by BentoML server."""

    config: Config
    inner: type[T]

    bento: t.Optional[Bento] = attrs.field(init=False, default=None)
    models: list[Model] = attrs.field(factory=list)
    apis: dict[str, APIMethod[..., t.Any]] = attrs.field(factory=dict)
    dependencies: dict[str, Dependency[t.Any]] = attrs.field(factory=dict, init=False)
    startup_hooks: list[LifecycleHook] = attrs.field(factory=list, init=False)
    shutdown_hooks: list[LifecycleHook] = attrs.field(factory=list, init=False)
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

        for field in dir(self.inner):
            value = getattr(self.inner, field)
            if isinstance(value, Dependency):
                self.dependencies[field] = t.cast(Dependency[t.Any], value)
            elif isinstance(value, Model):
                self.models.append(value)
            elif isinstance(value, APIMethod):
                self.apis[field] = t.cast("APIMethod[..., t.Any]", value)

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
    def find_dependent(self, name_or_path: str) -> Service[t.Any]:
        """Find a service by name or path"""
        attr_name, _, path = name_or_path.partition(".")
        if attr_name not in self.dependencies:
            if attr_name in self.all_services():
                return self.all_services()[attr_name]
            else:
                raise ValueError(f"Service {attr_name} not found")
        if path:
            return self.dependencies[attr_name].on.find_dependent(path)
        return self.dependencies[attr_name].on

    @lru_cache(maxsize=1)
    def all_services(self) -> dict[str, Service[t.Any]]:
        """Get a map of the service and all recursive dependencies"""
        services: dict[str, Service[t.Any]] = {self.name: self}
        for dependency in self.dependencies.values():
            services.update(dependency.on.all_services())
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

    def on_startup(self, func: HookF_ctx) -> HookF_ctx:
        self.startup_hooks.append(partial(func, self.context))
        return func

    def on_shutdown(self, func: HookF_ctx) -> HookF_ctx:
        self.shutdown_hooks.append(partial(func, self.context))
        return func

    def mount_asgi_app(
        self, app: ext.ASGIApp, path: str = "/", name: str | None = None
    ) -> None:
        self.mount_apps.append((app, path, name))  # type: ignore

    def mount_wsgi_app(
        self, app: ext.WSGIApp, path: str = "/", name: str | None = None
    ) -> None:
        # TODO: Migrate to a2wsgi
        from starlette.middleware.wsgi import WSGIMiddleware

        self.mount_apps.append((WSGIMiddleware(app), path, name))  # type: ignore

    def add_asgi_middleware(
        self, middleware_cls: type[ext.AsgiMiddleware], **options: t.Any
    ) -> None:
        self.middlewares.append((middleware_cls, options))

    def __call__(self) -> T:
        try:
            return self.inner()
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
    ) -> None:
        from _bentoml_impl.server import serve_http
        from bentoml._internal.log import configure_logging

        configure_logging()

        serve_http(
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
        )


@t.overload
def service(inner: type[T], /) -> Service[T]:
    ...


@t.overload
def service(inner: None = ..., /, **kwargs: Unpack[Config]) -> _ServiceDecorator:
    ...


def service(inner: type[T] | None = None, /, **kwargs: Unpack[Config]) -> t.Any:
    """Mark a class as a BentoML service.

    Example:

        @service(traffic={"timeout": 60})
        class InferenceService:
            @api
            def predict(self, input: str) -> str:
                return input
    """
    config = validate(kwargs)

    def decorator(inner: type[T]) -> Service[T]:
        if isinstance(inner, Service):
            raise TypeError("service() decorator can only be applied once")
        return Service(config=config, inner=inner)

    return decorator(inner) if inner is not None else decorator


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
            name=method.name,
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
    return Service(config=config, inner=RunnerHandle, models=runner.models, apis=apis)
