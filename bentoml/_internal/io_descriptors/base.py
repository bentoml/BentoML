from __future__ import annotations

import typing as t
from abc import ABCMeta
from abc import abstractmethod
from typing import overload
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from types import UnionType

    from typing_extensions import Self
    from starlette.requests import Request
    from starlette.responses import Response

    from bentoml.grpc.types import ProtoField
    from bentoml.grpc.types import BentoServicerContext
    from bentoml.grpc.v1.service_pb2 import Part as GRPCPart
    from bentoml.grpc.v1.service_pb2 import Request as GRPCRequest
    from bentoml.grpc.v1.service_pb2 import Response as GRPCResponse

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


class DescriptorMeta(ABCMeta):
    _proto_field: str

    def __new__(
        cls: type[Self],
        name: str,
        bases: tuple[type, ...],
        namespace: dict[str, t.Any],
        *,
        proto_field: str | None = None,
        **kwargs: t.Any,
    ) -> Self:
        if not proto_field:
            proto_field = ""

        klass = super().__new__(cls, name, bases, namespace, **kwargs)
        klass._proto_field = proto_field

        return klass


class IODescriptor(t.Generic[IOType], metaclass=DescriptorMeta, proto_field=None):
    """
    IODescriptor describes the input/output data format of an InferenceAPI defined
    in a :code:`bentoml.Service`. This is an abstract base class for extending new HTTP
    endpoint IO descriptor types in BentoServer.
    """

    HTTP_METHODS = ["POST"]

    _proto_field: str
    _mime_type: str
    _rpc_content_type: str

    def __new__(  # pylint: disable=unused-argument
        cls: t.Type[Self],
        *args: t.Any,
        **kwargs: t.Any,
    ) -> Self:
        self = super().__new__(cls)

        # default mime type is application/json
        self._mime_type = "application/json"
        # default grpc content type is application/grpc
        self._rpc_content_type = "application/grpc"

        return self

    def __repr__(self) -> str:
        return self.__class__.__qualname__

    @property
    def proto_field(self) -> ProtoField:
        """
        Returns a list of kinds fields that the IODescriptor can accept.

        Make sure to keep in sync with bentoml.grpc.v1.Request message.
        """
        return t.cast("ProtoField", self._proto_field)

    @property
    def grpc_content_type(self) -> str:
        generic_content_type = ["application/octet-stream", "text/plain"]
        if self._mime_type in generic_content_type or self._mime_type.startswith(
            "multipart"
        ):
            return f"{self._rpc_content_type}+proto"

        return f"{self._rpc_content_type}+{self._mime_type.split('/')[-1]}"

    @grpc_content_type.setter
    def grpc_content_type(self, value: str) -> None:
        self._rpc_content_type = value

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

    @overload
    @abstractmethod
    async def from_grpc_request(
        self, request: GRPCRequest, context: BentoServicerContext
    ) -> IOType:
        ...

    @overload
    @abstractmethod
    async def from_grpc_request(
        self, request: GRPCPart, context: BentoServicerContext
    ) -> IOType:
        ...

    @abstractmethod
    async def from_grpc_request(
        self, request: GRPCRequest | GRPCPart, context: BentoServicerContext
    ) -> IOType:
        ...

    @abstractmethod
    async def to_grpc_response(
        self, obj: IOType, context: BentoServicerContext
    ) -> GRPCResponse:
        ...

    # TODO: add generate_protobuf(self)
    # to generate protobuf from python object
