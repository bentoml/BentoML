from __future__ import annotations

import abc
import typing as t

if t.TYPE_CHECKING:
    from starlette.requests import Request
    from starlette.responses import Response


class Serde(abc.ABC):
    @abc.abstractmethod
    async def serialize_request(self, request: Request) -> bytes:
        raise NotImplementedError

    @abc.abstractmethod
    async def deserialize_request(self, data: bytes) -> Request:
        raise NotImplementedError

    @abc.abstractmethod
    async def serialize_response(self, response: Response) -> bytes:
        raise NotImplementedError

    @abc.abstractmethod
    async def deserialize_response(self, data: bytes) -> Response:
        raise NotImplementedError


class JSONSerde(Serde):
    # TODO: Implement JSONSerde
    async def serialize_request(self, request: Request) -> bytes:
        raise NotImplementedError

    async def deserialize_request(self, data: bytes) -> Request:
        raise NotImplementedError

    async def serialize_response(self, response: Response) -> bytes:
        raise NotImplementedError

    async def deserialize_response(self, data: bytes) -> Response:
        raise NotImplementedError
