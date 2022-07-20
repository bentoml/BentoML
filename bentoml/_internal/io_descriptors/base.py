from __future__ import annotations

import typing as t
from abc import ABC
from abc import abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from types import UnionType

    import grpc
    from starlette.requests import Request
    from starlette.responses import Response

    from ..types import LazyType
    from ...protos import service_pb2
    from ..context import InferenceApiContext as Context

    InputType = (
        UnionType
        | t.Type[t.Any]
        | LazyType[t.Any]
        | dict[str, t.Type[t.Any] | UnionType | LazyType[t.Any]]
    )


IOPyObj = t.TypeVar("IOPyObj")


_T = t.TypeVar("_T")


class IODescriptor(ABC, t.Generic[IOPyObj]):
    """
    IODescriptor describes the input/output data format of an InferenceAPI defined
    in a :code:`bentoml.Service`. This is an abstract base class for extending new HTTP
    endpoint IO descriptor types in BentoServer.
    """

    HTTP_METHODS = ["POST"]
    _init_str: str = ""

    def __new__(cls: t.Type[_T], *args: t.Any, **kwargs: t.Any) -> _T:
        self = super().__new__(cls)
        arg_strs = tuple(repr(i) for i in args) + tuple(
            f"{k}={repr(v)}" for k, v in kwargs.items()
        )
        setattr(self, "_init_str", f"{cls.__name__}({', '.join(arg_strs)})")

        return self

    def __repr__(self) -> str:
        return self._init_str

    @abstractmethod
    def input_type(self) -> InputType:
        ...

    @abstractmethod
    def openapi_schema_type(self) -> dict[str, str]:
        ...

    @abstractmethod
    def openapi_request_schema(self) -> dict[str, t.Any]:
        ...

    @abstractmethod
    def openapi_responses_schema(self) -> dict[str, t.Any]:
        ...

    @abstractmethod
    async def from_http_request(self, request: Request) -> IOPyObj:
        ...

    @abstractmethod
    async def to_http_response(
        self, obj: IOPyObj, ctx: Context | None = None
    ) -> Response:
        ...

    @abstractmethod
    def generate_protobuf(self):
        ...

    @abstractmethod
    async def from_grpc_request(
        self, request: service_pb2.RouteCallRequest, context: grpc.ServicerContext
    ) -> IOPyObj:
        ...

    @abstractmethod
    async def to_grpc_response(self, obj: IOPyObj) -> service_pb2.RouteCallResponse:
        ...
