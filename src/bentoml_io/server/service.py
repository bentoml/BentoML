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

from bentoml._internal.configuration.containers import BentoMLContainer
from bentoml._internal.context import ServiceContext
from bentoml.exceptions import BentoMLException

from ..client import ClientManager
from ..servable import Servable

if t.TYPE_CHECKING:
    from bentoml._internal import external_typing as ext
    from bentoml._internal.types import LifecycleHook

    ContextFunc = t.Callable[[ServiceContext], None | t.Coroutine[t.Any, t.Any, None]]
    HookF = t.TypeVar("HookF", bound=LifecycleHook)
    HookF_ctx = t.TypeVar("HookF_ctx", bound=ContextFunc)


@attrs.define
class Service:
    servable_cls: type[Servable]
    args: tuple[t.Any, ...] = attrs.field(default=(), repr=False)
    kwargs: dict[str, t.Any] = attrs.field(factory=dict, repr=False)
    dependencies: list[Service] = attrs.field(factory=list)
    mount_apps: list[tuple[ext.ASGIApp, str, str]] = attrs.field(
        factory=list, init=False, repr=False
    )
    middlewares: list[tuple[type[ext.AsgiMiddleware], dict[str, t.Any]]] = attrs.field(
        factory=list, init=False, repr=False
    )
    startup_hooks: list[LifecycleHook] = attrs.field(
        factory=list, init=False, repr=False
    )
    shutdown_hooks: list[LifecycleHook] = attrs.field(
        factory=list, init=False, repr=False
    )
    # service context
    context: ServiceContext = attrs.field(
        init=False, factory=ServiceContext, repr=False
    )
    # import info
    _caller_module: str = attrs.field(init=False, repr=False)
    _working_dir: str = attrs.field(init=False, repr=False, factory=os.getcwd)
    _servable: Servable | None = attrs.field(init=False, repr=False, default=None)
    _client_manager: ClientManager = attrs.field(init=False, repr=False)

    def __attrs_post_init__(self):
        self.startup_hooks.append(self.init_servable)  # type: ignore

        async def cleanup() -> None:
            await self._client_manager.cleanup()
            self.destroy_servable()

        self.shutdown_hooks.append(cleanup)

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

    @cached_property
    def import_string(self) -> str:
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
                return f"{import_module}:{name}"
        raise BentoMLException(
            "Failed to get service import origin, bentoml.Service object must be assigned to a variable at module level"
        )

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
    def name(self) -> str:
        return self.servable_cls.name

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
        from .serving import serve_http_production

        if working_dir is None:
            working_dir = os.getcwd()
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
