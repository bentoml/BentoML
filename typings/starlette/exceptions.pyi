"""
This type stub file was generated by pyright.
"""

import typing
from starlette.requests import Request
from starlette.responses import Response
from starlette.types import ASGIApp, Receive, Scope, Send

class HTTPException(Exception):
    def __init__(self, status_code: int, detail: str = ...) -> None: ...
    def __repr__(self) -> str: ...

class ExceptionMiddleware:
    def __init__(
        self, app: ASGIApp, handlers: dict = ..., debug: bool = ...
    ) -> None: ...
    def add_exception_handler(
        self,
        exc_class_or_status_code: typing.Union[int, typing.Type[Exception]],
        handler: typing.Callable,
    ) -> None: ...
    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None: ...
    def http_exception(self, request: Request, exc: HTTPException) -> Response: ...
