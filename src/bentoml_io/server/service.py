from __future__ import annotations

import inspect
import os
import sys
import typing as t
from functools import cached_property
from functools import partial

import attrs
from simple_di import Provide
from simple_di import inject

from bentoml._internal.bento.bento import Bento
from bentoml._internal.configuration.containers import BentoMLContainer
from bentoml._internal.context import ServiceContext
from bentoml._internal.runner.strategy import DefaultStrategy
from bentoml._internal.runner.strategy import Strategy
from bentoml._internal.utils import dict_filter_none
from bentoml._internal.utils import first_not_none
from bentoml.exceptions import BentoMLException

from ..api import APIMethod
from ..client import ClientManager
from ..servable import Servable

if t.TYPE_CHECKING:
    from bentoml._internal import external_typing as ext
    from bentoml._internal.types import LifecycleHook

    ContextFunc = t.Callable[[ServiceContext], None | t.Coroutine[t.Any, t.Any, None]]
    HookF = t.TypeVar("HookF", bound=LifecycleHook)
    HookF_ctx = t.TypeVar("HookF_ctx", bound=ContextFunc)

    class BatchingConfig(t.TypedDict, total=False):
        max_batch_size: int
        max_latency_ms: int


@attrs.define(slots=False)
class Service:
    servable_cls: type[Servable]
    name: str = attrs.field()
    args: tuple[t.Any, ...] = attrs.field(default=())
    kwargs: dict[str, t.Any] = attrs.field(factory=dict)
    dependencies: list[Service] = attrs.field(factory=list)
    bento: Bento | None = None
    max_batch_size: int | None = attrs.field(default=None, kw_only=True)
    max_latency_ms: int | None = attrs.field(default=None, kw_only=True)
    method_configs: dict[str, BatchingConfig] = attrs.field(factory=dict, kw_only=True)
    strategy: type[Strategy] = attrs.field(default=DefaultStrategy, kw_only=True)

    service_methods: dict[str, APIMethod[..., t.Any]] = attrs.field(
        init=False, factory=dict
    )
    mount_apps: list[tuple[ext.ASGIApp, str, str]] = attrs.field(
        factory=list, init=False
    )
    middlewares: list[tuple[type[ext.AsgiMiddleware], dict[str, t.Any]]] = attrs.field(
        factory=list, init=False
    )
    startup_hooks: list[LifecycleHook] = attrs.field(factory=list, init=False)
    shutdown_hooks: list[LifecycleHook] = attrs.field(factory=list, init=False)
    # service context
    context: ServiceContext = attrs.field(init=False, factory=ServiceContext)
    # import info
    _caller_module: str = attrs.field(init=False)
    _working_dir: str = attrs.field(init=False, factory=os.getcwd)
    _import_str: str | None = attrs.field(init=False, default=None)
    _servable: Servable | None = attrs.field(init=False, default=None)
    _client_manager: ClientManager = attrs.field(init=False)

    def __attrs_post_init__(self):
        self.startup_hooks.append(self.init_servable)  # type: ignore

        async def cleanup() -> None:
            await self._client_manager.cleanup()
            self.destroy_servable()

        self.shutdown_hooks.append(cleanup)

        config = BentoMLContainer.runners_config.get()
        if not isinstance(self, APIService) and self.name in config:
            config = config[self.name]

        for name, method in self.servable_cls.__servable_methods__.items():
            method_config = self.method_configs.get(name, {})
            new_config = dict_filter_none(
                dict(
                    max_batch_size=first_not_none(
                        method_config.get("max_batch_size"),
                        self.max_batch_size,
                        config["batching"]["max_batch_size"],
                    ),
                    max_latency_ms=first_not_none(
                        method_config.get("max_latency_ms"),
                        self.max_latency_ms,
                        config["batching"]["max_latency_ms"],
                    ),
                ),
            )
            self.service_methods[name] = attrs.evolve(method, **new_config)

    def __repr__(self) -> str:
        return f"<{self.__class__.__name__} name={self.name}>"

    @_caller_module.default
    def get_caller_module(self) -> str:
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

    @_client_manager.default
    def get_client_manager(self) -> ClientManager:
        return ClientManager(self)

    @name.default
    def get_name(self) -> str:
        return self.servable_cls.name

    @property
    def import_string(self) -> str:
        if self._import_str is None:
            import_module = self._caller_module
            if import_module == "__main__":
                if hasattr(sys.modules["__main__"], "__file__"):
                    import_module = sys.modules["__main__"].__file__
                else:
                    raise BentoMLException(
                        "Failed to get service import origin, bentoml.Service object defined interactively in console or notebook is not supported"
                    )

            if self._caller_module not in sys.modules:
                raise BentoMLException(
                    "Failed to get service import origin, bentoml.Service object must be defined in a module"
                )

            for name, value in vars(sys.modules[self._caller_module]).items():
                if value is self:
                    self._import_str = f"{import_module}:{name}"
                    break
            else:
                raise BentoMLException(
                    "Failed to get service import origin, bentoml.Service object must be assigned to a variable at module level"
                )
        return self._import_str

    def add_dependencies(self, *services: Service) -> None:
        self.dependencies.extend(services)

    def mount_asgi_app(
        self, app: "ext.ASGIApp", path: str = "/", name: str | None = None
    ) -> None:
        self.mount_apps.append((app, path, name))  # type: ignore

    def mount_wsgi_app(
        self, app: ext.WSGIApp, path: str = "/", name: str | None = None
    ) -> None:
        # TODO: Migrate to a2wsgi
        from starlette.middleware.wsgi import WSGIMiddleware

        self.mount_apps.append((WSGIMiddleware(app), path, name))  # type: ignore

    def add_asgi_middleware(
        self, middleware_cls: t.Type[ext.AsgiMiddleware], **options: t.Any
    ) -> None:
        self.middlewares.append((middleware_cls, options))

    @property
    def servable(self) -> Servable:
        if self._servable is None:
            raise BentoMLException("Service is not initialized")
        return self._servable

    def init_servable(self) -> Servable:
        if self._servable is None:
            self._servable = self.servable_cls(*self.args, **self.kwargs)
            self._servable.get_client = self._client_manager.get_client
        return self._servable

    def destroy_servable(self) -> None:
        self._servable = None

    def on_startup(self, func: HookF_ctx) -> HookF_ctx:
        self.startup_hooks.append(partial(func, self.context))
        return func

    def on_shutdown(self, func: HookF_ctx) -> HookF_ctx:
        self.shutdown_hooks.append(partial(func, self.context))
        return func

    @cached_property
    def config(self) -> dict[str, t.Any]:
        config = BentoMLContainer.runners_config.get()
        if self.name in config:
            return config[self.name]
        return config

    @property
    def worker_count(self) -> int:
        config = self.config
        return self.strategy.get_worker_count(
            self.servable_cls,
            config["resources"],
            config.get("workers_per_resource", 1),
        )

    @property
    def worker_env_map(self) -> list[dict[str, t.Any]]:
        config = self.config
        return [
            self.strategy.get_worker_env(
                self.servable_cls,
                config["resources"],
                config.get("workers_per_resource", 1),
                i,
            )
            for i in range(self.worker_count)
        ]

    @inject
    def serve_http(
        self,
        *,
        working_dir: str | None = None,
        port: int = Provide[BentoMLContainer.http.port],
        host: str = Provide[BentoMLContainer.http.host],
        backlog: int = Provide[BentoMLContainer.api_server_config.backlog],
        api_workers: int = Provide[BentoMLContainer.api_server_workers],
        timeout: int | None = None,
        ssl_certfile: str | None = Provide[BentoMLContainer.ssl.certfile],
        ssl_keyfile: str | None = Provide[BentoMLContainer.ssl.keyfile],
        ssl_keyfile_password: str
        | None = Provide[BentoMLContainer.ssl.keyfile_password],
        ssl_version: int | None = Provide[BentoMLContainer.ssl.version],
        ssl_cert_reqs: int | None = Provide[BentoMLContainer.ssl.cert_reqs],
        ssl_ca_certs: str | None = Provide[BentoMLContainer.ssl.ca_certs],
        ssl_ciphers: str | None = Provide[BentoMLContainer.ssl.ciphers],
        bentoml_home: str = Provide[BentoMLContainer.bentoml_home],
        development_mode: bool = False,
        reload: bool = False,
    ) -> None:
        from bentoml._internal.log import configure_logging

        from .serving import serve_http_production

        configure_logging()

        if working_dir is None:
            working_dir = self._working_dir
        serve_http_production(
            self,
            working_dir=working_dir,
            host=host,
            port=port,
            backlog=backlog,
            api_workers=api_workers,
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


class APIService(Service):
    """A service that doesn't scale on requested resources"""

    @property
    def worker_count(self) -> int:
        return BentoMLContainer.api_server_workers.get()

    @property
    def worker_env_map(self) -> list[dict[str, t.Any]]:
        return []
