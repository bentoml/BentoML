from __future__ import annotations

import typing as t
from abc import ABC
from abc import abstractmethod
from typing import TYPE_CHECKING

from starlette.requests import Request
from starlette.responses import Response

if TYPE_CHECKING:
    from types import UnionType

    from ..types import LazyType
    from ..context import InferenceApiContext as Context


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
    def input_type(
        self,
    ) -> t.Union[
        "UnionType",
        t.Type[t.Any],
        "LazyType[t.Any]",
        t.Dict[str, t.Union[t.Type[t.Any], "UnionType", "LazyType[t.Any]"]],
    ]:
        ...

    @abstractmethod
    def openapi_schema_type(self) -> t.Dict[str, str]:
        ...

    @abstractmethod
    def openapi_request_schema(self) -> t.Dict[str, t.Any]:
        ...

    @abstractmethod
    def openapi_responses_schema(self) -> t.Dict[str, t.Any]:
        ...

    @abstractmethod
    async def from_http_request(self, request: Request) -> IOPyObj:
        ...

    @abstractmethod
    async def to_http_response(
        self, obj: IOPyObj, ctx: Context | None = None
    ) -> Response:
        ...

    # TODO: gRPC support
    # @abstractmethod
    # def generate_protobuf(self): ...

    # @abstractmethod
    # async def from_grpc_request(self, request: GRPCRequest) -> IOType: ...

    # @abstractmethod
    # async def to_grpc_response(self, obj: IOType) -> GRPCResponse: ...
