

import sys
import types
import typing
from enum import Enum

from starlette.convertors import Convertor
from starlette.datastructures import URLPath
from starlette.types import ASGIApp, Receive, Scope, Send

if sys.version_info >= (3, 7): ...
else: ...

class NoMatchFound(Exception):
    """
    Raised by `.url_for(name, **path_params)` and `.url_path_for(name, **path_params)`
    if no matching route exists.
    """

    ...

class Match(Enum):
    NONE = ...
    PARTIAL = ...
    FULL = ...

def iscoroutinefunction_or_partial(obj: typing.Any) -> bool:
    """
    Correctly determines if an object is a coroutine function,
    including those wrapped in functools.partial objects.
    """
    ...

def request_response(func: typing.Callable) -> ASGIApp:
    """
    Takes a function or coroutine `func(request) -> response`,
    and returns an ASGI application.
    """
    ...

def websocket_session(func: typing.Callable) -> ASGIApp:
    """
    Takes a coroutine `func(session)`, and returns an ASGI application.
    """
    ...

def get_name(endpoint: typing.Callable) -> str: ...
def replace_params(
    path: str,
    param_convertors: typing.Dict[str, Convertor],
    path_params: typing.Dict[str, str],
) -> typing.Tuple[str, dict]: ...

PARAM_REGEX = ...

def compile_path(
    path: str,
) -> typing.Tuple[typing.Pattern, str, typing.Dict[str, Convertor]]:
    """
    Given a path string, like: "/{username:str}", return a three-tuple
    of (regex, format, {param_name:convertor}).

    regex:      "/(?P<username>[^/]+)"
    format:     "/{username}"
    convertors: {"username": StringConvertor()}
    """
    ...

class BaseRoute:
    def matches(self, scope: Scope) -> typing.Tuple[Match, Scope]: ...
    def url_path_for(self, name: str, **path_params: str) -> URLPath: ...
    async def handle(self, scope: Scope, receive: Receive, send: Send) -> None: ...
    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        """
        A route may be used in isolation as a stand-alone ASGI app.
        This is a somewhat contrived case, as they'll almost always be used
        within a Router, but could be useful for some tooling and minimal apps.
        """
        ...

class Route(BaseRoute):
    def __init__(
        self,
        path: str,
        endpoint: typing.Callable,
        *,
        methods: typing.List[str] = ...,
        name: str = ...,
        include_in_schema: bool = ...
    ) -> None: ...
    def matches(self, scope: Scope) -> typing.Tuple[Match, Scope]: ...
    def url_path_for(self, name: str, **path_params: str) -> URLPath: ...
    async def handle(self, scope: Scope, receive: Receive, send: Send) -> None: ...
    def __eq__(self, other: typing.Any) -> bool: ...

class WebSocketRoute(BaseRoute):
    def __init__(
        self, path: str, endpoint: typing.Callable, *, name: str = ...
    ) -> None: ...
    def matches(self, scope: Scope) -> typing.Tuple[Match, Scope]: ...
    def url_path_for(self, name: str, **path_params: str) -> URLPath: ...
    async def handle(self, scope: Scope, receive: Receive, send: Send) -> None: ...
    def __eq__(self, other: typing.Any) -> bool: ...

class Mount(BaseRoute):
    def __init__(
        self,
        path: str,
        app: ASGIApp = ...,
        routes: typing.Sequence[BaseRoute] = ...,
        name: str = ...,
    ) -> None: ...
    @property
    def routes(self) -> typing.List[BaseRoute]: ...
    def matches(self, scope: Scope) -> typing.Tuple[Match, Scope]: ...
    def url_path_for(self, name: str, **path_params: str) -> URLPath: ...
    async def handle(self, scope: Scope, receive: Receive, send: Send) -> None: ...
    def __eq__(self, other: typing.Any) -> bool: ...

class Host(BaseRoute):
    def __init__(self, host: str, app: ASGIApp, name: str = ...) -> None: ...
    @property
    def routes(self) -> typing.List[BaseRoute]: ...
    def matches(self, scope: Scope) -> typing.Tuple[Match, Scope]: ...
    def url_path_for(self, name: str, **path_params: str) -> URLPath: ...
    async def handle(self, scope: Scope, receive: Receive, send: Send) -> None: ...
    def __eq__(self, other: typing.Any) -> bool: ...

_T = ...

class _AsyncLiftContextManager(typing.AsyncContextManager[_T]):
    def __init__(self, cm: typing.ContextManager[_T]) -> None: ...
    async def __aenter__(self) -> _T: ...
    async def __aexit__(
        self,
        exc_type: typing.Optional[typing.Type[BaseException]],
        exc_value: typing.Optional[BaseException],
        traceback: typing.Optional[types.TracebackType],
    ) -> typing.Optional[bool]: ...

class _DefaultLifespan:
    def __init__(self, router: Router) -> None: ...
    async def __aenter__(self) -> None: ...
    async def __aexit__(self, *exc_info: object) -> None: ...
    def __call__(self: _T, app: object) -> _T: ...

class Router:
    def __init__(
        self,
        routes: typing.Sequence[BaseRoute] = ...,
        redirect_slashes: bool = ...,
        default: ASGIApp = ...,
        on_startup: typing.Sequence[typing.Callable] = ...,
        on_shutdown: typing.Sequence[typing.Callable] = ...,
        lifespan: typing.Callable[[typing.Any], typing.AsyncContextManager] = ...,
    ) -> None: ...
    async def not_found(self, scope: Scope, receive: Receive, send: Send) -> None: ...
    def url_path_for(self, name: str, **path_params: str) -> URLPath: ...
    async def startup(self) -> None:
        """
        Run any `.on_startup` event handlers.
        """
        ...
    async def shutdown(self) -> None:
        """
        Run any `.on_shutdown` event handlers.
        """
        ...
    async def lifespan(self, scope: Scope, receive: Receive, send: Send) -> None:
        """
        Handle ASGI lifespan messages, which allows us to manage application
        startup and shutdown events.
        """
        ...
    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        """
        The main entry point to the Router class.
        """
        ...
    def __eq__(self, other: typing.Any) -> bool: ...
    def mount(self, path: str, app: ASGIApp, name: str = ...) -> None: ...
    def host(self, host: str, app: ASGIApp, name: str = ...) -> None: ...
    def add_route(
        self,
        path: str,
        endpoint: typing.Callable,
        methods: typing.List[str] = ...,
        name: str = ...,
        include_in_schema: bool = ...,
    ) -> None: ...
    def add_websocket_route(
        self, path: str, endpoint: typing.Callable, name: str = ...
    ) -> None: ...
    def route(
        self,
        path: str,
        methods: typing.List[str] = ...,
        name: str = ...,
        include_in_schema: bool = ...,
    ) -> typing.Callable: ...
    def websocket_route(self, path: str, name: str = ...) -> typing.Callable: ...
    def add_event_handler(self, event_type: str, func: typing.Callable) -> None: ...
    def on_event(self, event_type: str) -> typing.Callable: ...
