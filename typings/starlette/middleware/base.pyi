import typing
from starlette.requests import Request
from starlette.responses import Response
from starlette.types import ASGIApp, Receive, Scope, Send

RequestResponseEndpoint = typing.Callable[[Request], typing.Awaitable[Response]]
DispatchFunction = typing.Callable[
    [Request, RequestResponseEndpoint], typing.Awaitable[Response]
]

class BaseHTTPMiddleware:
    def __init__(self, app: ASGIApp, dispatch: DispatchFunction = ...) -> None: ...
    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None: ...
    async def dispatch(
        self, request: Request, call_next: RequestResponseEndpoint
    ) -> Response: ...
