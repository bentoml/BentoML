import contextlib
from concurrent import futures
from typing import Callable, Dict, Iterable, Optional, Union, List

import grpc
from prometheus_client import start_http_server

from bentoml.yatai.proto.mock_service_pb2 import MockRequest, MockResponse
from bentoml.yatai.proto.mock_service_pb2_grpc import MockServiceServicer
from bentoml.utils import reserve_free_port

SpecialCaseFunction = Callable[[str, grpc.ServicerContext], str]


# this would also work for future interceptor if needed
class MockService(MockServiceServicer):
    """A gRPC service used for testing

    Args:
        special_cases: dict where keys are string, values are servicer_context
        that take and return strings. This will allow testing exception"""

    def __init__(self, special_cases: Union[str, Dict[str, SpecialCaseFunction]]):
        self._special_cases = special_cases if isinstance(special_cases, dict) else ""

    def Execute(self, request: MockRequest, context:grpc.ServicerContext) -> MockResponse:
        return MockResponse(output=self.__get_output(request, context))

    def ExecuteClientStream(
        self, request_iterator: Iterable[MockRequest], context: grpc.ServicerContext
    ) -> MockResponse:
        output = "".join(
            self.__get_output(request, context) for request in request_iterator
        )
        return MockResponse(output=output)

    def ExecuteServerStream(
        self, request: MockRequest, context: grpc.ServicerContext
    ) -> Iterable[MockResponse]:
        for _o in self.__get_output(request, context):
            yield MockResponse(output=_o)

    def ExecuteClientServerStream(
        self, request_iterator: Iterable[MockRequest], context: grpc.ServicerContext
    ) -> Iterable[MockResponse]:
        for request in request_iterator:
            yield MockResponse(output=self.__get_output(request, context))

    def __get_output(self, request: MockRequest, context: grpc.ServicerContext) -> str:
        inp = request.input
        output = inp

        if isinstance(self._special_cases, dict):
            if inp in self._special_cases:
                output = self._special_cases[inp][inp, context]

        return output


class MockServerClient:
    def __init__(
        self,
        special_cases: Union[str, Dict[str, SpecialCaseFunction]],
        server_interceptors: Optional[List] = None,
        prometheus_enabled: Optional[bool] = True,
    ):
        self.special_cases = special_cases
        self.server_interceptors = server_interceptors
        self.prometheus_enabled = prometheus_enabled
        self.mock_server: grpc.Server
        with reserve_free_port() as server_port:
            self.server_port: int =  server_port
        with reserve_free_port() as prom_port:
            self.prom_port: int = prom_port

    @contextlib.contextmanager
    def mock_server_client(self):
        """A context manager returns a gRPC client connected with MockService"""
        from bentoml.yatai.proto.mock_service_pb2_grpc import (
            add_MockServiceServicer_to_server,
            MockServiceStub,
        )

        if not self.server_interceptors:
            interceptors = []
        else:
            interceptors = self.server_interceptors

        self.mock_server = grpc.server(
            futures.ThreadPoolExecutor(max_workers=1), interceptors=interceptors
        )
        mock_service = MockService(special_cases=self.special_cases)
        add_MockServiceServicer_to_server(mock_service, self.mock_server)

        # reserve a free port for windows, else we use unix domain socket

        self.mock_server.add_insecure_port(f'[::]:{self.server_port}')
        self.mock_server.start()

        if self.prometheus_enabled:
            start_http_server(self.prom_port)
        channel = grpc.insecure_channel(f'localhost:{self.server_port}')

        client_stub = MockServiceStub(channel)

        try:
            yield client_stub
        finally:
            self.mock_server.stop(None)
