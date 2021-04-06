from unittest import TestCase
from unittest.mock import Mock
from parameterized import parameterized

from bentoml.yatai.client.interceptor.prom_server_interceptor import (
    PromServerInterceptor,
    ServiceLatencyInterceptor,
    _wrap_rpc_behaviour,
)
from bentoml.yatai.utils import parse_method_name, MethodName

import requests

from tests.yatai.grpc_testing_service import MockRequest, MockServerClient


def test_method_name():
    # Fields are correct and fully_qualified_service work.
    mn = MethodName("foo.bar", "SearchService", "Search")
    assert mn.package == "foo.bar"
    assert mn.service == "SearchService"
    assert mn.method == "Search"
    assert mn.fully_qualified_service == "foo.bar.SearchService"


def test_empty_package_method_name():
    # fully_qualified_service works when there's no package
    mn = MethodName("", "SearchService", "Search")
    assert mn.fully_qualified_service == "SearchService"


def test_parse_method_name():
    mn, ok = parse_method_name("/foo.bar.SearchService/Search")
    assert mn.package == "foo.bar"
    assert mn.service == "SearchService"
    assert mn.method == "Search"
    assert ok


def test_parse_empty_package():
    # parse_method_name works with no package.
    mn, _ = parse_method_name("/SearchService/Search")
    assert mn.package == ""
    assert mn.service == "SearchService"
    assert mn.method == "Search"


class TestPrometheusServerInterceptor(TestCase):
    def setUp(self):
        self.interceptor: PromServerInterceptor = PromServerInterceptor()

    def test_handler_none(self):
        mock_continuation = Mock(return_value=None)

        ret = self.interceptor.intercept_service(
            mock_continuation, Mock(return_value=None)
        )

        assert not ret

    @parameterized.expand(
        [(True, True, 1), (True, False, 1), (False, True, 1), (False, False, 2)],
    )
    def test_intercept_service_unary(
        self, request_streaming, response_streaming, calls
    ):
        mock_handler_call_details = Mock(method="/grpc-service")
        mock_handler = Mock(
            request_streaming=request_streaming, response_streaming=response_streaming
        )
        mock_continuation = Mock(return_value=mock_handler)

        self.interceptor.intercept_service(mock_continuation, mock_handler_call_details)
        assert calls == mock_continuation.call_count

    @parameterized.expand([("/grpc-service"), ("/grpc-service/grpc-method")])
    def test_intercept_service(self, method):
        mock_handler_call_details = Mock(method=method)
        mock_handler = Mock(request_streaming=False, response_streaming=False)
        mock_continuation = Mock(return_value=mock_handler)

        self.interceptor.intercept_service(mock_continuation, mock_handler_call_details)
        assert 2 == mock_continuation.call_count


class TestServiceLatencyInterceptor(TestCase):
    def setUp(self):
        self.interceptor: ServiceLatencyInterceptor = ServiceLatencyInterceptor()

    @parameterized.expand([("/grpc-latency"), ("/grpc-latency/grpc-method")])
    def test_intercept_service(self, method):
        mock_handler_call_details = Mock(method=method)
        mock_handler = Mock(request_streaming=False, response_streaming=False)
        mock_continuation = Mock(return_value=mock_handler)

        self.interceptor.intercept_service(mock_continuation, mock_handler_call_details)
        assert mock_continuation.call_count == 2


class TestWrapRPCBehaviour(TestCase):
    @parameterized.expand(
        [
            (True, True, "stream_stream"),
            (True, False, "stream_unary"),
            (False, True, "unary_stream"),
            (False, False, "unary_unary"),
        ]
    )
    def test_wrap_rpc_behaviour(self, request_streaming, response_streaming, behaviour):
        mock_handler = Mock(
            request_streaming=request_streaming, response_streaming=response_streaming
        )
        mock_fn = Mock()
        res = _wrap_rpc_behaviour(mock_handler, mock_fn)
        assert mock_fn.call_count == 1
        assert getattr(res, behaviour) is not None

    def test_wrap_rpc_behaviour_none(self):
        assert not _wrap_rpc_behaviour(None, Mock())


class TestMetrics(TestCase):
    @parameterized.expand(
        [
            (
                'grpc_server_started_total\
                {grpc_method="Execute",\
                grpc_service="bentoml.MockService",\
                grpc_type="UNARY"}',
                1.0,
            ),
            (
                'grpc_server_started_total\
                {grpc_method="Execute",\
                grpc_service="bentoml.MockService",\
                grpc_type="UNARY"}',
                2.0,
            ),
            (
                'grpc_server_started_total\
                {grpc_method="Execute",\
                grpc_service="bentoml.MockService",\
                grpc_type="UNARY"}',
                3.0,
            ),
            (
                'grpc_server_handled_total\
                {grpc_code="OK",\
                grpc_method="Execute",\
                grpc_service="bentoml.MockService",\
                grpc_type="UNARY"}',
                4.0,
            ),
        ]
    )
    def test_grpc_server_metrics(self, metric_name, value):
        mock_server = MockServerClient(
            special_cases="",
            server_interceptors=[PromServerInterceptor(), ServiceLatencyInterceptor()],
        )
        with mock_server.mock_server_client() as client:
            assert client.Execute(MockRequest(input="foo")).output == "foo"
            r = requests.get(f"http://localhost:{mock_server.prom_port}")
            assert (
                f"{metric_name} {value}" in r.text
            ), f"expected metrics {metric_name} {value}\
                not found in server response:\n{r.text}"
