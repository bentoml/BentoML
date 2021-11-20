"""
This type stub file was generated by pyright.
"""

import typing as t
from collections.abc import Mapping

from starlette.datastructures import URL, Address, FormData, Headers, QueryParams, State
from starlette.types import Message, Receive, Scope, Send

SERVER_PUSH_HEADERS_TO_COPY = ...

def cookie_parser(cookie_string: str) -> t.Dict[str, str]:
    """
    This function parses a ``Cookie`` HTTP header into a dict of key/value pairs.

    It attempts to mimic browser cookie parsing behavior: browsers and web servers
    frequently disregard the spec (RFC 6265) when setting and reading cookies,
    so we attempt to suit the common scenarios here.

    This function has been adapted from Django 3.1.0.
    Note: we are explicitly _NOT_ using `SimpleCookie.load` because it is based
    on an outdated spec and will fail on lots of input we want to support
    """
    ...

class ClientDisconnect(Exception): ...

class HTTPConnection(Mapping):
    """
    A base class for incoming HTTP connections, that is used to provide
    any functionality that is common to both `Request` and `WebSocket`.
    """

    scope: Scope = ...
    def __init__(self, scope: Scope, receive: Receive = ...) -> None: ...
    def __getitem__(self, key: str) -> str: ...
    def __iter__(self) -> t.Iterator[str]: ...
    def __len__(self) -> int: ...
    __eq__ = ...
    __hash__ = ...
    @property
    def app(self) -> t.Any: ...
    @property
    def url(self) -> URL: ...
    @property
    def base_url(self) -> URL: ...
    @property
    def headers(self) -> Headers: ...
    @property
    def query_params(self) -> QueryParams: ...
    @property
    def path_params(self) -> dict: ...
    @property
    def cookies(self) -> t.Dict[str, str]: ...
    @property
    def client(self) -> Address: ...
    @property
    def session(self) -> dict: ...
    @property
    def auth(self) -> t.Any: ...
    @property
    def user(self) -> t.Any: ...
    @property
    def state(self) -> State: ...
    def url_for(self, name: str, **path_params: t.Any) -> str: ...

async def empty_receive() -> Message: ...
async def empty_send(message: Message) -> None: ...

class Request(HTTPConnection):
    _body: bytes = ...
    def __init__(
        self, scope: Scope, receive: Receive = ..., send: Send = ...
    ) -> None: ...
    @property
    def method(self) -> str: ...
    @property
    def receive(self) -> Receive: ...
    async def stream(self) -> t.AsyncGenerator[bytes, None]: ...
    async def body(self) -> bytes: ...
    async def json(self) -> t.Any: ...
    async def form(self) -> FormData: ...
    async def close(self) -> None: ...
    async def is_disconnected(self) -> bool: ...
    async def send_push_promise(self, path: str) -> None: ...
