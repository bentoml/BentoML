from __future__ import annotations

import typing as t
from abc import ABCMeta
from abc import abstractmethod
from typing import TYPE_CHECKING

from starlette.requests import Request
from starlette.responses import Response

if TYPE_CHECKING:
    from types import UnionType

    from typing_extensions import Self

    from bentoml.grpc.v1.service_pb2 import Request as GRPCRequest
    from bentoml.grpc.v1.service_pb2 import Response as GRPCResponse

    from ..types import LazyType
    from ..context import InferenceApiContext as Context
    from ..server.grpc.types import BentoServicerContext
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


class DescriptorMeta(ABCMeta):
    def __new__(
        cls: type[Self],
        name: str,
        bases: tuple[type, ...],
        namespace: dict[str, t.Any],
        *,
        proto_fields: list[str] | None = None,
    ) -> Self:
        if not proto_fields:
            proto_fields = []
        namespace["_proto_fields"] = proto_fields
        return super().__new__(cls, name, bases, namespace)


class IODescriptor(t.Generic[IOType], metaclass=DescriptorMeta, proto_fields=None):
    """
    IODescriptor describes the input/output data format of an InferenceAPI defined
    in a :code:`bentoml.Service`. This is an abstract base class for extending new HTTP
    endpoint IO descriptor types in BentoServer.
    """

    HTTP_METHODS = ["POST"]

    _init_str: str = ""
    _proto_fields: list[str]

    _mime_type: str

    def __new__(cls: t.Type[Self], *args: t.Any, **kwargs: t.Any) -> Self:
        self = super().__new__(cls)
        # default mime type is application/json
        self._mime_type = "application/json"
        self._init_str = cls.__qualname__

        return self

    def __repr__(self) -> str:
        return self._init_str

    @property
    def accepted_proto_fields(self) -> list[str]:
        """
        Returns a list of kinds fields that the IODescriptor can accept.

        Make sure to keep in sync with bentoml.grpc.v1.Value message.
        """
        return self._proto_fields

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
    def generate_protobuf(self):
        ...

    @abstractmethod
    async def from_grpc_request(
        self, request: GRPCRequest, context: BentoServicerContext
    ) -> IOType:
        ...

    @abstractmethod
    async def to_grpc_response(
        self, obj: IOType, context: BentoServicerContext
    ) -> GRPCResponse:
        ...
