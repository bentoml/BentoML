import unittest
import unittest.mock as mock

from bentoml.yatai.client.interceptor.prom_server_interceptor import (
    PromServerInterceptor,
)
import bentoml.yatai.utils as yatai_utils


class TestPrometheusServerInterceptor(unittest.TestCase):
    def test_constructor(self):
        bool_val = True
        self.assertFalse(
            PromServerInterceptor().enable_handling_time_historgram, bool_val
        )
        self.assertTrue(
            PromServerInterceptor(
                enable_handling_time_historgram=True
            ).enable_handling_time_historgram,
            bool_val,
        )
    
    def test_intercept_service_no_metadata(self):
        patch = mock.patch('bentoml.yatai.client.interceptor.prom_server_interceptor')
        mock_context = mock.Mock()
        mock_context.invocation_metatdata = mock.Mock(return_value=None)
        mock_context._rpc_event.call_details.method='hello'
        interceptor = PromServerInterceptor()
        mock_handler = mock.Mock()
        mock_handler.request_streaming = False
        mock_handler.response_streaming = False
        mock_continuation = mock.Mock(return_value=mock_handler)

        with patch:
            interceptor.intercept_service(mock_continuation, mock.Mock()).unary_unary(mock.Mock(), mock_context)
