"""
@generated by mypy-protobuf.  Do not edit manually!
isort:skip_file
"""
import abc
import bentoml.grpc.v1.service_pb2
import grpc

class BentoServiceStub:
    """a gRPC BentoServer."""

    def __init__(self, channel: grpc.Channel) -> None: ...
    Call: grpc.UnaryUnaryMultiCallable[
        bentoml.grpc.v1.service_pb2.Request,
        bentoml.grpc.v1.service_pb2.Response,
    ]
    """Call handles methodcaller of given API entrypoint."""
    ServiceMetadata: grpc.UnaryUnaryMultiCallable[
        bentoml.grpc.v1.service_pb2.ServiceMetadataRequest,
        bentoml.grpc.v1.service_pb2.ServiceMetadataResponse,
    ]
    """ServiceMetadata returns metadata of bentoml.Service."""

class BentoServiceServicer(metaclass=abc.ABCMeta):
    """a gRPC BentoServer."""

    @abc.abstractmethod
    def Call(
        self,
        request: bentoml.grpc.v1.service_pb2.Request,
        context: grpc.ServicerContext,
    ) -> bentoml.grpc.v1.service_pb2.Response:
        """Call handles methodcaller of given API entrypoint."""
    @abc.abstractmethod
    def ServiceMetadata(
        self,
        request: bentoml.grpc.v1.service_pb2.ServiceMetadataRequest,
        context: grpc.ServicerContext,
    ) -> bentoml.grpc.v1.service_pb2.ServiceMetadataResponse:
        """ServiceMetadata returns metadata of bentoml.Service."""

def add_BentoServiceServicer_to_server(servicer: BentoServiceServicer, server: grpc.Server) -> None: ...
