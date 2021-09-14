from abc import ABC, abstractmethod
from typing import TypeVar

from ..types import HTTPRequest, HTTPResponse

IOPyObj = TypeVar("IOPyObj")


class IODescriptor(ABC):
    """IODescriptor describes the input/output data format of an InferenceAPI defined
    in a bentoml.Service
    """

    @abstractmethod
    def openapi_request_schema(self):
        pass

    @abstractmethod
    def openapi_responses_schema(self):
        pass

    @abstractmethod
    async def from_http_request(self, request: HTTPRequest) -> IOPyObj:
        pass

    @abstractmethod
    async def to_http_response(self, obj: IOPyObj) -> HTTPResponse:
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
