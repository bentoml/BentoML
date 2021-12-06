"""
This type stub file was generated by pyright.
"""

import enum
import typing

from starlette.requests import HTTPConnection
from starlette.types import Message, Receive, Scope, Send

class WebSocketState(enum.Enum):
    CONNECTING = ...
    CONNECTED = ...
    DISCONNECTED = ...

class WebSocketDisconnect(Exception):
    def __init__(self, code: int = ...) -> None: ...

class WebSocket(HTTPConnection):
    def __init__(self, scope: Scope, receive: Receive, send: Send) -> None: ...
    async def receive(self) -> Message:
        """
        Receive ASGI websocket messages, ensuring valid state transitions.
        """
        ...
    async def send(self, message: Message) -> None:
        """
        Send ASGI websocket messages, ensuring valid state transitions.
        """
        ...
    async def accept(self, subprotocol: str = ...) -> None: ...
    async def receive_text(self) -> str: ...
    async def receive_bytes(self) -> bytes: ...
    async def receive_json(self, mode: str = ...) -> typing.Any: ...
    async def iter_text(self) -> typing.AsyncIterator[str]: ...
    async def iter_bytes(self) -> typing.AsyncIterator[bytes]: ...
    async def iter_json(self) -> typing.AsyncIterator[typing.Any]: ...
    async def send_text(self, data: str) -> None: ...
    async def send_bytes(self, data: bytes) -> None: ...
    async def send_json(self, data: typing.Any, mode: str = ...) -> None: ...
    async def close(self, code: int = ...) -> None: ...

class WebSocketClose:
    def __init__(self, code: int = ...) -> None: ...
    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None: ...
