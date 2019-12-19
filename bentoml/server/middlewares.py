from timeit import default_timer

from flask import Request

from bentoml import config


class InstrumentMiddleware:
    def __init__(self, app, bento_service):
        self.app = app
        self.bento_service = bento_service

        from prometheus_client import Histogram, Counter, Gauge

        service_name = self.bento_service.name
        namespace = config('instrument').get('default_namespace')

        self.metrics_request_duration = Histogram(
            name=service_name + '_request_duration_seconds',
            documentation=service_name + " API HTTP request duration in seconds",
            namespace=namespace,
            labelnames=['endpoint', 'service_version', 'http_response_code'],
        )
        self.metrics_request_total = Counter(
            name=service_name + "_request_total",
            documentation='Totoal number of HTTP requests',
            namespace=namespace,
            labelnames=['endpoint', 'service_version', 'http_response_code'],
        )
        self.metrics_request_in_progress = Gauge(
            name=service_name + "_request_in_progress",
            documentation='Totoal number of HTTP requests in progress now',
            namespace=namespace,
            labelnames=['endpoint', 'service_version'],
        )

    def __call__(self, environ, start_response):
        req = Request(environ)
        endpoint = req.path
        start_time = default_timer()

        def start_response_wrapper(status, headers):
            ret = start_response(status, headers)
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
