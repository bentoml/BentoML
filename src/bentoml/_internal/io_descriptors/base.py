from __future__ import annotations

import typing as t
from abc import ABC
from abc import abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from types import UnionType

    from starlette.requests import Request
    from starlette.responses import Response

    from bentoml.grpc.types import ProtoField

    from ..types import LazyType
    from ..context import InferenceApiContext as Context
    from ..service.openapi.specification import Schema
    from ..service.openapi.specification import Response as OpenAPIResponse
    from ..service.openapi.specification import Reference
    from ..service.openapi.specification import RequestBody

    InputType = (
        UnionType
        | t.Type[t.Any]
        | LazyType[t.Any]
        | dict[str, t.Type[t.Any] | UnionType | LazyType[t.Any]]
    )


IOType = t.TypeVar("IOType")


class IODescriptor(ABC, t.Generic[IOType]):
    """
    IODescriptor describes the input/output data format of an InferenceAPI defined
    in a :code:`bentoml.Service`. This is an abstract base class for extending new HTTP
    endpoint IO descriptor types in BentoServer.
    """

    HTTP_METHODS = ["POST"]

    _mime_type: str
    _rpc_content_type: str = "application/grpc"
    _proto_fields: tuple[ProtoField]

    def __repr__(self) -> str:
        return self.__class__.__qualname__

    @abstractmethod
    def input_type(self) -> InputType:
        ...

    @abstractmethod
    def openapi_schema(self) -> Schema | Reference:
        raise NotImplementedError

    @abstractmethod
    def openapi_components(self) -> dict[str, t.Any] | None:
        raise NotImplementedError

    @abstractmethod
    def openapi_request_body(self) -> RequestBody:
        raise NotImplementedError

    @abstractmethod
    def openapi_responses(self) -> OpenAPIResponse:
        raise NotImplementedError

    @abstractmethod
    async def from_http_request(self, request: Request) -> IOType:
        ...

    @abstractmethod
    async def to_http_response(
        self, obj: IOType, ctx: Context | None = None
    ) -> Response:
        ...

    @abstractmethod
    async def from_proto(self, field: t.Any) -> IOType:
        ...

    @abstractmethod
    async def to_proto(self, obj: IOType) -> t.Any:
        ...
