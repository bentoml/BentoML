from __future__ import annotations

import typing as t
from abc import ABCMeta
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

    class OpenAPI:
        schema: Schema | Reference
        parameters: Parameter | Reference
        requestBody: RequestBody
        responses: OpenAPIResponse


IOType = t.TypeVar("IOType")


def simplify(obj: t.Any) -> str:
    if isinstance(obj, type):
        # We only need __qualname__ instead of repr
        return obj.__qualname__
    elif isinstance(obj, str):
        return obj
    return repr(obj)


def to_camel_case(name: str) -> str:
    comp = name.split("_")
    return comp[0] + "".join(x.title() for x in comp[1:])


class DescriptorMeta(ABCMeta):
    def __new__(
        cls: type,
        name: str,
        bases: tuple[type, ...],
        namespace: dict[str, t.Any],
    ) -> None:
        def openapi_namespace(prefix: str = "openapi") -> tuple[str]:
            return tuple(
                x[len(prefix) + 2 :] for x in dir(cls) if x.startswith(f"_{prefix}_")
            )

        openapi_attrs = {
            to_camel_case(k): property(
                fget=getattr(cls, f"_openapi_{k}"), doc=f"OpenAPI {k} object."
            )
            for k in openapi_namespace()
        }
        setattr(cls, "openapi", type("OpenAPI", (), openapi_attrs))

        return super().__new__(cls, name, bases, namespace)


class IODescriptor(t.Generic[IOType], metaclass=DescriptorMeta):
    """
    IODescriptor describes the input/output data format of an InferenceAPI defined
    in a :code:`bentoml.Service`. This is an abstract base class for extending new HTTP
    endpoint IO descriptor types in BentoServer.
    """

    HTTP_METHODS = ["POST"]

    _init_str: str = ""

    _mime_type: str
    openapi: OpenAPI

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
    def _openapi_schema(self) -> Schema | Reference:
        raise NotImplementedError

    @abstractmethod
    def _openapi_parameters(self) -> Parameter | Reference:
        raise NotImplementedError

    @abstractmethod
    def _openapi_components(self) -> Components:
        raise NotImplementedError

    @abstractmethod
    def _openapi_request_body(self) -> RequestBody:
        raise NotImplementedError

    @abstractmethod
    def _openapi_responses(self) -> OpenAPIResponse:
        raise NotImplementedError

    @abstractmethod
    async def from_http_request(self, request: Request) -> IOType:
        ...

    @abstractmethod
    async def to_http_response(
        self, obj: IOType, ctx: Context | None = None
    ) -> Response:
        ...

    # TODO: gRPC support
    # @abstractmethod
    # def generate_protobuf(self): ...

    # @abstractmethod
    # async def from_grpc_request(self, request: GRPCRequest) -> IOType: ...

    # @abstractmethod
    # async def to_grpc_response(self, obj: IOType) -> GRPCResponse: ...
