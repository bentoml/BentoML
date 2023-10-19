from __future__ import annotations

import inspect
import os
import sys
import typing as t
from functools import cached_property
from functools import partial

import attrs

from bentoml._internal.context import ServiceContext
from bentoml.exceptions import BentoMLException

from ..servable import Servable

if t.TYPE_CHECKING:
    from bentoml._internal import external_typing as ext
    from bentoml._internal.types import LifecycleHook

    ContextFunc = t.Callable[[ServiceContext], None | t.Coroutine[t.Any, t.Any, None]]
    HookF = t.TypeVar("HookF", bound=LifecycleHook)
    HookF_ctx = t.TypeVar("HookF_ctx", bound=ContextFunc)
    Dependency = t.Union["Service", str]


@attrs.define
class Service:
    servable_cls: type[Servable]
    args: tuple[t.Any, ...] = attrs.field(default=(), repr=False)
    kwargs: dict[str, t.Any] = attrs.field(factory=dict, repr=False)
    dependencies: list[Dependency] = attrs.field(factory=list)
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

    def add_dependencies(self, *services: Dependency) -> None:
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

    def get_servable(self) -> Servable:
        return self.servable_cls(*self.args, **self.kwargs)

    def on_startup(self, func: HookF_ctx) -> HookF_ctx:
        self.startup_hooks.append(partial(func, self.context))
        return func

    def on_shutdown(self, func: HookF_ctx) -> HookF_ctx:
        self.shutdown_hooks.append(partial(func, self.context))
        return func

    def start(self) -> None:
        raise NotImplementedError()  # TODO


if __name__ == "__main__":
    from .. import api

    class Foo(Servable):
        @api
        def greet(self, name: str) -> str:
            return f"Hello {name}"

    class Bar(Servable):
        @api
        def greet(self, name: str) -> str:
            return f"Hello {name}"

    svc = Service(Foo, dependencies=[Service(Bar)])

    @svc.on_startup
    def startup(ctx):
        pass
