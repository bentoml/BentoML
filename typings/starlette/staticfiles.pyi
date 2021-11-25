

import os
import typing

from starlette.datastructures import Headers
from starlette.responses import Response
from starlette.types import Receive, Scope, Send

PathLike = typing.Union[str, "os.PathLike[str]"]

class NotModifiedResponse(Response):
    NOT_MODIFIED_HEADERS = ...
    def __init__(self, headers: Headers) -> None: ...

class StaticFiles:
    def __init__(
        self,
        *,
        directory: PathLike = ...,
        packages: typing.List[str] = ...,
        html: bool = ...,
        check_dir: bool = ...
    ) -> None: ...
    def get_directories(
        self, directory: PathLike = ..., packages: typing.List[str] = ...
    ) -> typing.List[PathLike]:
        """
        Given `directory` and `packages` arguments, return a list of all the
        directories that should be used for serving static files from.
        """
        ...
    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        """
        The ASGI entry point.
        """
        ...
    def get_path(self, scope: Scope) -> str:
        """
        Given the ASGI scope, return the `path` string to serve up,
        with OS specific path seperators, and any '..', '.' components removed.
        """
        ...
    async def get_response(self, path: str, scope: Scope) -> Response:
        """
        Returns an HTTP response, given the incoming path, method and request headers.
        """
        ...
    async def lookup_path(
        self, path: str
    ) -> typing.Tuple[str, typing.Optional[os.stat_result]]: ...
    def file_response(
        self,
        full_path: PathLike,
        stat_result: os.stat_result,
        scope: Scope,
        status_code: int = ...,
    ) -> Response: ...
    async def check_config(self) -> None:
        """
        Perform a one-off configuration check that StaticFiles is actually
        pointed at a directory, so that we can raise loud errors rather than
        just returning 404 responses.
        """
        ...
    def is_not_modified(
        self, response_headers: Headers, request_headers: Headers
    ) -> bool:
        """
        Given the request and response headers, return `True` if an HTTP
        "Not Modified" response could be returned instead.
        """
        ...
