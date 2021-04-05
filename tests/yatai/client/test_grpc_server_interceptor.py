from unittest import TestCase
from unittest.mock import Mock

from bentoml.yatai.client.interceptor.prom_server_interceptor import (
    PromServerInterceptor,
    ServiceLatencyInterceptor, _wrap_rpc_behaviour
)

import grpc
import pytest
import requests

from tests.yatai.grpc_testing_service import mock_client, MockRequest, MockService


class TestPrometheusServerInterceptor(TestCase):
    def setUp(self):
        self.interceptor: PromServerInterceptor = PromServerInterceptor()

    @pytest.mark.parametrize(
        "request_streaming,response_streaming,calls",
        [(True, True, 1), (True, False, 1), (False, True, 1), (False, False, 2)],
    )
    def test_intercept_service_unary(
        self, request_streaming, response_streaming, calls
    ):
        mock_handler_call_details = Mock(method="/MockService")
        mock_handler = Mock(
            request_streaming=request_streaming, response_streaming=response_streaming
        )
        mock_continuation = Mock(return_value=mock_handler)

        self.interceptor.intercept_service(mock_continuation, mock_handler_call_details)
        assert calls == mock_continuation.call_count

    @pytest.mark.parametrize("method", [("/MockService"), ("/MockService/MockMethod")])
    def test_intercept_service_too_short(self, method):
        mock_handler_call_details = Mock(method=method)
        mock_handler = Mock(request_streaming=False, response_streaming=False)
        mock_continuation = Mock(return_value=mock_handler)

        self.interceptor.intercept_service(mock_continuation, mock_handler_call_details)
        assert mock_continuation.call_count == 2

    def test_non_handler(self):
        mock_continuation = Mock(return_value=None)

        ret = self.interceptor.intercept_service(mock_continuation, Mock())

        assert ret == None


class TestServiceLatencyInterceptor(TestCase):
    def setUp(self):
        self.interceptor: ServiceLatencyInterceptor = ServiceLatencyInterceptor()

    @pytest.mark.parametrize(
        "method", [("/grpc-service-only"), ("/grpc-service/grpc-method"),]
    )
    def test_intercept_service_method_name_too_short(self, method):
        mock_handler_call_details = Mock(method=method)
        mock_handler = Mock(request_streaming=False, response_streaming=False)
        mock_continuation = Mock(return_value=mock_handler)

        self.interceptor.intercept_service(mock_continuation, mock_handler_call_details)
        assert mock_continuation.call_count == 1

class Test_wrap_rpc_behaviour(TestCase):

    @pytest.mark.parametrize(
        "request_streaming,response_streaming,behaviour",
        [(True, True, "stream_stream"),
        (True, False, "stream_unary"),
        (False, True, "unary_stream"),
        (False, False, "unary_unary"),]
    )
    def test_wrap_rpc_behaviour(self, request_streaming, response_streaming, behaviour):
        mock_handler = Mock(request_streaming=request_streaming, response_streaming=response_streaming)
        mock_fn = Mock()
        res = _wrap_rpc_behaviour(mock_handler, mock_fn)
        assert mock_fn.call_count == 1
        assert getattr(res, behaviour) is not None

    def test_wrap_rpc_behaviour_non(self):
        assert _wrap_rpc_behaviour(None, Mock()) == None


