import unittest
import unittest.mock as mock

from bentoml.yatai.client.interceptor.prom_server_interceptor import PromServerInterceptor
import bentoml.yatai.utils as yatai_utils

class TestPrometheusServerInterceptor(unittest.TestCase):
    def test_constructor(self):
        bool_val = True
        self.assertFalse(PromServerInterceptor().enable_handling_time_historgram,bool_val)
        self.assertTrue(PromServerInterceptor(enable_handling_time_historgram=True).enable_handling_time_historgram, bool_val)

    def test


