import inspect
import typing as t
from abc import ABC, abstractmethod

from starlette.requests import Request
from starlette.responses import Response

IOPyObj = t.TypeVar("IOPyObj")


def _mk_str(obj: t.Any) -> str:
    # make str more human readable
    if callable(obj):
        return obj.__name__
    elif inspect.isclass(obj):
        return obj.__class__.__name__
    elif isinstance(obj, dict):
        fac = dict()  # type: t.Dict[str, t.Any]
        fac.update(zip(obj.keys(), map(_mk_str, obj.values())))  # type: ignore
        return str(fac)
    else:
        return str(obj)


class IODescriptor(ABC, t.Generic[IOPyObj]):
    """
    IODescriptor describes the input/output data format of an InferenceAPI defined
     in a bentoml.Service
    """

    HTTP_METHODS = ["POST"]

    def __str__(self) -> str:
        return f"%s(%s)" % (
            self.__class__.__name__,
            ", ".join(
                [f'{k.strip("_")}={_mk_str(v)}' for k, v in self.__dict__.items()]
            ),
        )

    # fmt: off
    @abstractmethod
    def openapi_schema_type(self) -> t.Dict[str, str]: ...  # noqa: E704

    @abstractmethod
    def openapi_request_schema(self) -> t.Dict[str, t.Any]: ...  # noqa: E704

    @abstractmethod
    def openapi_responses_schema(self) -> t.Dict[str, t.Any]: ...  # noqa: E704

    @abstractmethod
    async def from_http_request(self, request: Request) -> IOPyObj: ...  # noqa: E704

    @abstractmethod
    async def to_http_response(self, obj: IOPyObj) -> Response: ...  # noqa: E704

    # TODO: gRPC support
    # @abstractmethod
    # def generate_protobuf(self): ...  # noqa: E704

    # @abstractmethod
    # async def from_grpc_request(self, request: GRPCRequest) -> IOPyObj: ...

    # @abstractmethod
    # async def to_grpc_response(self, obj: IOPyObj) -> GRPCResponse: ...

    # fmt: on
