import os, typing as t
from starlette.background import BackgroundTask
from starlette.datastructures import URL, MutableHeaders
from starlette.types import Receive, Scope, Send

def guess_type(
    url: t.Union[str, os.PathLike[str]], strict: bool = ...
) -> t.Tuple[t.Optional[str], t.Optional[str]]: ...

class Response:
    media_type: str = ...
    charset: str = ...
    raw_headers: t.List[t.Tuple[bytes, bytes]] = ...
    body: bytes = ...
    def __init__(
        self,
        content: t.Any = ...,
        status_code: int = ...,
        headers: dict = ...,
        media_type: str = ...,
        background: BackgroundTask = ...,
    ) -> None: ...
    def render(self, content: t.Any) -> bytes: ...
    def init_headers(self, headers: t.Mapping[str, str] = ...) -> None: ...
    @property
    def headers(self) -> MutableHeaders: ...
    def set_cookie(
        self,
        key: str,
        value: str = ...,
        max_age: int = ...,
        expires: int = ...,
        path: str = ...,
        domain: str = ...,
        secure: bool = ...,
        httponly: bool = ...,
        samesite: str = ...,
    ) -> None: ...
    def delete_cookie(self, key: str, path: str = ..., domain: str = ...) -> None: ...
    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None: ...

class HTMLResponse(Response):
    media_type: str = ...

class PlainTextResponse(Response):
    media_type: str = ...

class JSONResponse(Response):
    media_type: str = ...
    def render(self, content: t.Any) -> bytes: ...

class RedirectResponse(Response):
    def __init__(
        self,
        url: t.Union[str, URL],
        status_code: int = ...,
        headers: dict = ...,
        background: BackgroundTask = ...,
    ) -> None: ...

class StreamingResponse(Response):
    def __init__(
        self,
        content: t.BinaryIO,
        status_code: int = ...,
        headers: t.Dict[bytes, bytes] = ...,
        media_type: str = ...,
        background: BackgroundTask = ...,
    ) -> None: ...
    async def listen_for_disconnect(self, receive: Receive) -> None: ...
    async def stream_response(self, send: Send) -> None: ...
    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None: ...

class FileResponse(Response):
    chunk_size: int = ...
    def __init__(
        self,
        path: t.Union[str, os.PathLike[str]],
        status_code: int = ...,
        headers: dict = ...,
        media_type: str = ...,
        background: BackgroundTask = ...,
        filename: str = ...,
        stat_result: os.stat_result = ...,
        method: str = ...,
    ) -> None: ...
    def set_stat_headers(self, stat_result: os.stat_result) -> None: ...
    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None: ...
