import contextvars
import multiprocessing
import os
import shutil
from timeit import default_timer

from flask import Request

from bentoml import config
from bentoml.server.utils import logger

from opentracing.ext import tags
from opentracing.propagation import Format
import opentracing
from jaeger_client import Config
import logging

request_span = contextvars.ContextVar('request_span')

class JaegerMiddleware:
    def __init__(self, app, bento_service):
        self.app = app
        self.bento_service = bento_service

        service_name = self.bento_service.name
        self.tracer = initialize_tracer('inference_pipeline')

    def __call__(self, environ, start_response):
        def start_response_wrapper(status, headers):
            ret = start_response(status, headers)
            status_code = int(status.split()[0])
            return ret

        # update trace id
        environ['uber-trace-id'] = environ['HTTP_UBER_TRACE_ID']
        span_ctx = self.tracer.extract(Format.HTTP_HEADERS, environ)
        request_method = environ['REQUEST_METHOD']
        path_info = environ['PATH_INFO']

        span = self.tracer.start_span(
            operation_name=f"{request_method} {path_info}",
            child_of=span_ctx
        )

        url = environ['werkzeug.request'].base_url
        span.set_tag(tags.HTTP_URL, url)

        remote_ip = environ['REMOTE_ADDR']
        span.set_tag(tags.PEER_HOST_IPV4, remote_ip or "")

        remote_port = environ['REMOTE_PORT']
        span.set_tag(tags.PEER_PORT, remote_port or "")

        span_tags = {tags.SPAN_KIND: tags.SPAN_KIND_RPC_SERVER}

        with self.tracer.scope_manager.activate(span, True) as scope:
            request_span.set(span)
            return self.app(environ, start_response_wrapper)


def initialize_tracer(service_name):
    logging.basicConfig(level=logging.DEBUG)

    config = Config(
        config={
            'sampler': {'type': 'const', 'param': 1},
            'logging': True
        }, service_name=service_name
    )

    tracer = config.initialize_tracer()
    if tracer is None:
        tracer = opentracing.global_tracer()

    return tracer
