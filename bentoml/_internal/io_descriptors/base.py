from __future__ import annotations

import typing as t
from abc import ABC
from abc import abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from types import UnionType

    from typing_extensions import Self
    from starlette.requests import Request
    from starlette.responses import Response

    from ..types import LazyType
    from ..context import InferenceApiContext as Context
    from ..service.openapi.specification import Schema
    from ..service.openapi.specification import Response as OpenAPIResponse
    from ..service.openapi.specification import Parameter
    from ..service.openapi.specification import Reference
    from ..service.openapi.specification import Components
    from ..service.openapi.specification import RequestBody

    InputType = (
        UnionType
        | t.Type[t.Any]
        | LazyType[t.Any]
        | dict[str, t.Type[t.Any] | UnionType | LazyType[t.Any]]
    )


IOType = t.TypeVar("IOType")


def simplify(obj: t.Any) -> str:
    if isinstance(obj, type):
        # We only need __qualname__ instead of repr
        return obj.__qualname__
    elif isinstance(obj, str):
        return obj
    return repr(obj)


class IODescriptor(ABC, t.Generic[IOType]):
    """
    IODescriptor describes the input/output data format of an InferenceAPI defined
    in a :code:`bentoml.Service`. This is an abstract base class for extending new HTTP
    endpoint IO descriptor types in BentoServer.
    """

    HTTP_METHODS = ["POST"]

    _init_str: str = ""

    _mime_type: str

    def __new__(cls: t.Type[Self], *args: t.Any, **kwargs: t.Any) -> Self:
        self = super().__new__(cls)
        arg_strs = tuple(map(simplify, args)) + tuple(
            f"{k}={simplify(v)}" for k, v in kwargs.items()
        )
        setattr(self, "_init_str", f"{cls.__name__}({','.join(arg_strs)})")

        return self

    def __repr__(self) -> str:
        return self._init_str

    @abstractmethod
    def input_type(self) -> InputType:
        ...

    @abstractmethod
    def openapi_schema(self) -> Schema | Reference:
        raise NotImplementedError

    @abstractmethod
    def openapi_parameter(self) -> Parameter | Reference:
        raise NotImplementedError

    @abstractmethod
    def openapi_components(self) -> Components:
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
