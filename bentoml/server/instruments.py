import multiprocessing
import os
import shutil
from timeit import default_timer

from flask import Request

from bentoml import config
from bentoml.server.utils import logger


class InstrumentMiddleware:
    def __init__(self, app, bento_service):
        self.app = app
        self.bento_service = bento_service

        from prometheus_client import Histogram, Counter, Gauge, CollectorRegistry

        service_name = self.bento_service.name
        namespace = config('instrument').get('default_namespace')
        # Use local registry instead of the global one to avoid duplicated metrics
        # register
        self.collector_registry = CollectorRegistry()

        self.metrics_request_duration = Histogram(
            name=service_name + '_request_duration_seconds',
            documentation=service_name + " API HTTP request duration in seconds",
            namespace=namespace,
            labelnames=['endpoint', 'service_version', 'http_response_code'],
            registry=self.collector_registry,
        )
        self.metrics_request_total = Counter(
            name=service_name + "_request_total",
            documentation='Total number of HTTP requests',
            namespace=namespace,
            labelnames=['endpoint', 'service_version', 'http_response_code'],
            registry=self.collector_registry,
        )
        self.metrics_request_in_progress = Gauge(
            name=service_name + "_request_in_progress",
            documentation='Total number of HTTP requests in progress now',
            namespace=namespace,
            labelnames=['endpoint', 'service_version'],
            registry=self.collector_registry,
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


def setup_prometheus_multiproc_dir(lock: multiprocessing.Lock = None):
    """
    Set up prometheus_multiproc_dir for prometheus to work in multiprocess mode,
    which is required when working with Gunicorn server

    Warning: for this to work, prometheus_client library must be imported after
    this function is called. It relies on the os.environ['prometheus_multiproc_dir']
    to properly setup for multiprocess mode
    """
    if lock is not None:
        lock.acquire()

    try:
        prometheus_multiproc_dir = config('instrument').get('prometheus_multiproc_dir')
        logger.debug(
            "Setting up prometheus_multiproc_dir: %s", prometheus_multiproc_dir
        )
        # Wipe prometheus metrics directory between runs
        # https://github.com/prometheus/client_python#multiprocess-mode-gunicorn
        # Ignore errors so it does not fail when directory does not exist
        shutil.rmtree(prometheus_multiproc_dir, ignore_errors=True)
        os.makedirs(prometheus_multiproc_dir, exist_ok=True)

        os.environ['prometheus_multiproc_dir'] = prometheus_multiproc_dir
    finally:
        if lock is not None:
            lock.release()
