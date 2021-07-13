import logging
import multiprocessing
from timeit import default_timer
from typing import TYPE_CHECKING

from simple_di import Provide, inject

from bentoml.configuration.containers import BentoMLContainer


if TYPE_CHECKING:
    Lock = multiprocessing.synchronize.Lock


logger = logging.getLogger(__name__)


class InstrumentMiddleware:
    @inject
    def __init__(
        self,
        app,
        bento_service,
        metrics_client=Provide[BentoMLContainer.metrics_client],
    ):
        self.app = app
        self.bento_service = bento_service

        service_name = self.bento_service.name

        self.metrics_request_duration = metrics_client.Histogram(
            name=service_name + '_request_duration_seconds',
            documentation=service_name + " API HTTP request duration in seconds",
            labelnames=['endpoint', 'service_version', 'http_response_code'],
        )
        self.metrics_request_total = metrics_client.Counter(
            name=service_name + "_request_total",
            documentation='Total number of HTTP requests',
            labelnames=['endpoint', 'service_version', 'http_response_code'],
        )
        self.metrics_request_in_progress = metrics_client.Gauge(
            name=service_name + "_request_in_progress",
            documentation='Total number of HTTP requests in progress now',
            labelnames=['endpoint', 'service_version'],
            multiprocess_mode='livesum',
        )

    def __call__(self, environ, start_response):
        from flask import Request

        req = Request(environ)
        endpoint = req.path
        start_time = default_timer()

        def start_response_wrapper(status, headers, exc_info=None):
            ret = start_response(status, headers, exc_info)
            status_code = int(status.split()[0])

            # instrument request total count
            self.metrics_request_total.labels(
                endpoint=endpoint,
                service_version=self.bento_service.version,
                http_response_code=status_code,
            ).inc()

            # instrument request duration
            total_time = max(default_timer() - start_time, 0)
            self.metrics_request_duration.labels(
                endpoint=endpoint,
                service_version=self.bento_service.version,
                http_response_code=status_code,
            ).observe(total_time)

            return ret

        with self.metrics_request_in_progress.labels(
            endpoint=endpoint, service_version=self.bento_service.version
        ).track_inprogress():
            return self.app(environ, start_response_wrapper)
