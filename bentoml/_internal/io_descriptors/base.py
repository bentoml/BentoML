from __future__ import annotations

import typing as t
from abc import ABC
from abc import abstractmethod
from typing import overload
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from types import UnionType

    from google.protobuf import message
    from google.protobuf import struct_pb2
    from google.protobuf import wrappers_pb2
    from typing_extensions import Self
    from starlette.requests import Request
    from starlette.responses import Response
    from google.protobuf.internal.containers import MessageMap

    from bentoml.grpc.types import ProtoField
    from bentoml.grpc.v1alpha1 import service_pb2 as pb

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
    _rpc_content_type: str
    _proto_field: str

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

    @property
    def accepted_proto_fields(self) -> ProtoField:
        """
        Returns a proto field that the IODescriptor can accept.
        Note that all proto field will also accept _internal_bytes_contents
        """
        return t.cast("ProtoField", self._proto_field)

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

    @overload
    @abstractmethod
    async def from_proto(
        self,
        field: wrappers_pb2.StringValue | pb.Part | bytes,
        *,
        _use_internal_bytes_contents: bool,
    ) -> IOType:
        ...

    @overload
    @abstractmethod
    async def from_proto(
        self,
        field: struct_pb2.Value | pb.Part | bytes,
        *,
        _use_internal_bytes_contents: bool,
    ) -> IOType:
        ...

    @overload
    @abstractmethod
    async def from_proto(
        self, field: MessageMap[str, pb.Part], *, _use_internal_bytes_contents: bool
    ) -> IOType:
        ...

    @overload
    @abstractmethod
    async def from_proto(
        self, field: pb.NDArray | pb.Part | bytes, *, _use_internal_bytes_contents: bool
    ) -> IOType:
        ...

    @overload
    @abstractmethod
    async def from_proto(
        self, field: pb.File | pb.Part | bytes, *, _use_internal_bytes_contents: bool
    ) -> IOType:
        ...

    @overload
    @abstractmethod
    async def from_proto(
        self,
        field: pb.DataFrame | pb.Part | bytes,
        *,
        _use_internal_bytes_contents: bool,
    ) -> IOType:
        ...

    @overload
    @abstractmethod
    async def from_proto(
        self, field: pb.Series | pb.Part | bytes, *, _use_internal_bytes_contents: bool
    ) -> IOType:
        ...

    @abstractmethod
    async def from_proto(
        self,
        field: message.Message | bytes | MessageMap[str, pb.Part],
        *,
        _use_internal_bytes_contents: bool = False,
    ) -> IOType:
        ...

    @abstractmethod
    async def to_proto(self, obj: IOType) -> MessageMap[str, pb.Part] | message.Message:
        ...
