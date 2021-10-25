import typing as t
from abc import ABC, abstractmethod

from starlette.requests import Request
from starlette.responses import Response


class IODescriptor(ABC):
    """IODescriptor describes the input/output data format of an InferenceAPI defined
    in a bentoml.Service
    """

    @abstractmethod
    def openapi_request_schema(self) -> t.Dict[str, t.Any]:
        pass

    @abstractmethod
    def openapi_responses_schema(self) -> t.Dict[str, t.Any]:
        pass

    @abstractmethod
    async def from_http_request(self, request: Request) -> t.Any:
        pass

    @abstractmethod
    async def to_http_response(self, obj: t.Any) -> Response:
        pass

    # TODO: gRPC support
    # @abstractmethod
    # def generate_protobuf(self):
    #     pass
    #
    # @abstractmethod
    # async def from_grpc_request(self, request: GRPCRequest) -> IOPyObj:
    #     pass
    #
    # @abstractmethod
    # async def to_grpc_response(self, obj: IOPyObj) -> GRPCResponse:
    #     pass
