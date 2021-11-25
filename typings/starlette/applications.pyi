

import typing

from starlette.datastructures import URLPath
from starlette.middleware import Middleware
from starlette.routing import BaseRoute
from starlette.types import ASGIApp, Receive, Scope, Send

class Starlette:
    """
    Creates an application instance.

    **Parameters:**

    * **debug** - Boolean indicating if debug tracebacks should be returned on errors.
    * **routes** - A list of routes to serve incoming HTTP and WebSocket requests.
    * **middleware** - A list of middleware to run for every request. A starlette
    application will always automatically include two middleware classes.
    `ServerErrorMiddleware` is added as the very outermost middleware, to handle
    any uncaught errors occurring anywhere in the entire stack.
    `ExceptionMiddleware` is added as the very innermost middleware, to deal
    with handled exception cases occurring in the routing or endpoints.
    * **exception_handlers** - A dictionary mapping either integer status codes,
    or exception class types onto callables which handle the exceptions.
    Exception handler callables should be of the form
    `handler(request, exc) -> response` and may be be either standard functions, or
    async functions.
    * **on_startup** - A list of callables to run on application startup.
    Startup handler callables do not take any arguments, and may be be either
    standard functions, or async functions.
    * **on_shutdown** - A list of callables to run on application shutdown.
    Shutdown handler callables do not take any arguments, and may be be either
    standard functions, or async functions.
    """

    def __init__(
        self,
        debug: bool = ...,
        routes: typing.Sequence[BaseRoute] = ...,
        middleware: typing.Sequence[Middleware] = ...,
        exception_handlers: typing.Dict[
            typing.Union[int, typing.Type[Exception]], typing.Callable
        ] = ...,
        on_startup: typing.Sequence[typing.Callable] = ...,
        on_shutdown: typing.Sequence[typing.Callable] = ...,
        lifespan: typing.Callable[[Starlette], typing.AsyncContextManager] = ...,
    ) -> None: ...
    def build_middleware_stack(self) -> ASGIApp: ...
    @property
    def routes(self) -> typing.List[BaseRoute]: ...
    @property
    def debug(self) -> bool: ...
    @debug.setter
    def debug(self, value: bool) -> None: ...
    def url_path_for(self, name: str, **path_params: str) -> URLPath: ...
    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None: ...
    def on_event(self, event_type: str) -> typing.Callable: ...
    def mount(self, path: str, app: ASGIApp, name: str = ...) -> None: ...
    def host(self, host: str, app: ASGIApp, name: str = ...) -> None: ...
    def add_middleware(self, middleware_class: type, **options: typing.Any) -> None: ...
    def add_exception_handler(
        self,
        exc_class_or_status_code: typing.Union[int, typing.Type[Exception]],
        handler: typing.Callable,
    ) -> None: ...
    def add_event_handler(self, event_type: str, func: typing.Callable) -> None: ...
    def add_route(
        self,
        path: str,
        route: typing.Callable,
        methods: typing.List[str] = ...,
        name: str = ...,
        include_in_schema: bool = ...,
    ) -> None: ...
    def add_websocket_route(
        self, path: str, route: typing.Callable, name: str = ...
    ) -> None: ...
    def exception_handler(
        self, exc_class_or_status_code: typing.Union[int, typing.Type[Exception]]
    ) -> typing.Callable: ...
    def route(
        self,
        path: str,
        methods: typing.List[str] = ...,
        name: str = ...,
        include_in_schema: bool = ...,
    ) -> typing.Callable: ...
    def websocket_route(self, path: str, name: str = ...) -> typing.Callable: ...
    def middleware(self, middleware_type: str) -> typing.Callable: ...
