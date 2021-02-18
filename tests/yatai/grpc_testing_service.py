import os
import contextlib
from concurrent import futures
from uuid import uuid4
from tempfile import gettempdir
from typing import Callable, Dict, Iterable, Optional, List

import grpc

from tests.yatai.proto import mock_service_pb2_grpc
from tests.yatai.proto.mock_service_pb2 import MockRequest, MockResponse
from bentoml.yatai.client.interceptor.prom_server_interceptor import PromServerInterceptor
from bentoml.utils import reserve_free_port

SpecialCaseFunction = Callable[[str, grpc.ServicerContext], str]

class MockService(mock_service_pb2_grpc.MockServiceServicer):
    """A gRPC service used for testing

    Args:
        special_cases: dict where keys are string, values are servicer_context that take and return strings
        This will allow testing exception"""

    def __init__(self, special_cases: Dict[str, SpecialCaseFunction]):
        self._special_cases = special_cases

    def Execute(self, request: MockRequest, context: grpc.ServicerContext) -> MockResponse:
        return MockResponse(outpu=self.__get_output(request, context))

    def ExecuteClientStream(self, request_iterator: Iterable[MockRequest], context: grpc.ServicerContext) -> MockResponse:
        output = "".join(self.__get_output(request, context) for request in request_iterator)
        return MockResponse(output=output)

    def ExecuteServerStream(self, request: MockRequest, context:grpc.ServicerContext) -> Iterable[MockResponse]:
        for _o in self.__get_output(request, context):
            yield MockResponse(output=_o)

    def ExecuteClientServerStream(self, request_iterator: Iterable[MockRequest], context: grpc.ServicerContext) -> Iterable[MockResponse]:
        for request in request_iterator:
            yield MockResponse(output=self.__get_output(request, context))

    def __get_output(self, request: MockRequest, context: grpc.ServicerContext) -> str:
        inp = request.input

        return self._special_cases[inp](inp,context) if inp in self._special_cases else inp


@contextlib.contextmanager
def mock_client(special_cases: Dict[str, SpecialCaseFunction], server_interceptor: Optional[List[PromServerInterceptor]]):
    """A context manager returns a gRPC client connected with MockService"""
    interceptors = [] if not server_interceptor else server_interceptor

    mock_server = grpc.server(futures.ThreadPoolExecutor(max_workers=1), interceptors=interceptors)
    mock_service = MockService(special_cases=special_cases)
    mock_service_pb2_grpc.add_MockServiceServicer_to_server(mock_service, mock_server)

    # We will initialize a free port and reserve it till context ends for windows, else we use unix domain socket
    with reserve_free_port() as mock_grpc_port:
        if os.name == 'nt':
            channel_descriptor = f'localhost:{mock_grpc_port}'
        else:
            channel_descriptor = f'unix://{gettempdir()}/{uuid4()}.sock'

    mock_server.add_insecure_port(channel_descriptor)
    mock_server.start()

    channel = grpc.insecure_channel(channel_descriptor)

    mock_client = mock_service_pb2_grpc.MockServiceStub(channel)

    try:
        yield mock_client
    finally:
        mock_server.stop(None)
