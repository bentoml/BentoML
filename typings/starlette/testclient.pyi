import sys, typing
from concurrent.futures import Future
from typing import TypedDict
import requests
from starlette.types import Message, Receive, Scope, Send

if sys.version_info >= (3, 8): ...
else: ...
_PortalFactoryType = typing.Callable[
    [], typing.ContextManager[anyio.abc.BlockingPortal]
]
Cookies = typing.Union[
    typing.MutableMapping[str, str], requests.cookies.RequestsCookieJar
]
Params = typing.Union[bytes, typing.MutableMapping[str, str]]
DataType = typing.Union[bytes, typing.MutableMapping[str, str], typing.IO]
TimeOut = typing.Union[float, typing.Tuple[float, float]]
FileType = typing.MutableMapping[str, typing.IO]
AuthType = (
    typing.Union[
        typing.Tuple[str, str],
        requests.auth.AuthBase,
        typing.Callable[[requests.PreparedRequest], requests.PreparedRequest],
    ],
)
ASGIInstance = typing.Callable[[Receive, Send], typing.Awaitable[None]]
ASGI2App = typing.Callable[[Scope], ASGIInstance]
ASGI3App = typing.Callable[[Scope, Receive, Send], typing.Awaitable[None]]

class _HeaderDict(requests.packages.urllib3._collections.HTTPHeaderDict):
    def get_all(self, key: str, default: str) -> str: ...

class _MockOriginalResponse:
    def __init__(self, headers: typing.List[typing.Tuple[bytes, bytes]]) -> None: ...
    def isclosed(self) -> bool: ...

class _Upgrade(Exception):
    def __init__(self, session: WebSocketTestSession) -> None: ...

class _WrapASGI2:
    def __init__(self, app: ASGI2App) -> None: ...
    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None: ...

class _AsyncBackend(TypedDict):
    backend: str
    backend_options: typing.Dict[str, typing.Any]
    ...

class _ASGIAdapter(requests.adapters.HTTPAdapter):
    def __init__(
        self,
        app: ASGI3App,
        portal_factory: _PortalFactoryType,
        raise_server_exceptions: bool = ...,
        root_path: str = ...,
    ) -> None: ...
    def send(
        self, request: requests.PreparedRequest, *args: typing.Any, **kwargs: typing.Any
    ) -> requests.Response: ...

class WebSocketTestSession:
    def __init__(
        self, app: ASGI3App, scope: Scope, portal_factory: _PortalFactoryType
    ) -> None: ...
    def __enter__(self) -> WebSocketTestSession: ...
    def __exit__(self, *args: typing.Any) -> None: ...
    def send(self, message: Message) -> None: ...
    def send_text(self, data: str) -> None: ...
    def send_bytes(self, data: bytes) -> None: ...
    def send_json(self, data: typing.Any, mode: str = ...) -> None: ...
    def close(self, code: int = ...) -> None: ...
    def receive(self) -> Message: ...
    def receive_text(self) -> str: ...
    def receive_bytes(self) -> bytes: ...
    def receive_json(self, mode: str = ...) -> typing.Any: ...

class TestClient(requests.Session):
    __test__ = ...
    task: Future[None]
    portal: typing.Optional[anyio.abc.BlockingPortal] = ...
    def __init__(
        self,
        app: typing.Union[ASGI2App, ASGI3App],
        base_url: str = ...,
        raise_server_exceptions: bool = ...,
        root_path: str = ...,
        backend: str = ...,
        backend_options: typing.Optional[typing.Dict[str, typing.Any]] = ...,
    ) -> None: ...
    def request(
        self,
        method: str,
        url: str,
        params: Params = ...,
        data: DataType = ...,
        headers: typing.MutableMapping[str, str] = ...,
        cookies: Cookies = ...,
        files: FileType = ...,
        auth: AuthType = ...,
        timeout: TimeOut = ...,
        allow_redirects: bool = ...,
        proxies: typing.MutableMapping[str, str] = ...,
        hooks: typing.Any = ...,
        stream: bool = ...,
        verify: typing.Union[bool, str] = ...,
        cert: typing.Union[str, typing.Tuple[str, str]] = ...,
        json: typing.Any = ...,
    ) -> requests.Response: ...
    def websocket_connect(
        self, url: str, subprotocols: typing.Sequence[str] = ..., **kwargs: typing.Any
    ) -> typing.Any: ...
    def __enter__(self) -> TestClient: ...
    def __exit__(self, *args: typing.Any) -> None: ...
    async def lifespan(self) -> None: ...
    async def wait_startup(self) -> None: ...
    async def wait_shutdown(self) -> None: ...
