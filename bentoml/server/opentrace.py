#!/usr/bin/python
# -*- coding: utf-8 -*-

from contextlib import contextmanager
from contextvars import ContextVar
import opentracing
import logging
from opentracing import Format
from functools import partial
from jaeger_client.config import Config
from opentracing.scope_managers.asyncio import AsyncioScopeManager

span_context_var = ContextVar('span context', default=None)


def initialize_tracer(service_name, log_level=logging.DEBUG):
    logging.basicConfig(level=log_level)

    config = Config(
        config={
            'sampler': {'type': 'const', 'param': 1},
            'logging': True
        },
        service_name=service_name,
        validate=True,
        scope_manager=AsyncioScopeManager()
    )

    return config.initialize_tracer()


@contextmanager
def trace(
        server_url=None,
        request_headers=None,
        async_transport=False,
        sample_rate=1.0,
        standalone=False,
        is_root=False,
        service_name="some service",
        span_name="service procedure",
        port=0,
):
    """
    Opentracing tracer function
    """

    tracer = initialize_tracer(service_name) or opentracing.global_tracer() or None
    if tracer is None:
        yield None
        return

    span_context = None
    span_context_saved = span_context_var.get()

    if request_headers is not None:
        span_context = tracer.extract(Format.HTTP_HEADERS, request_headers)

    if span_context is None:
        span_context = span_context_saved or None

    with tracer.start_active_span(operation_name=span_name,
                                  child_of=span_context) as scope:
        token = span_context_var.set(scope.span.context)
        yield scope
        span_context_var.reset(token)


async_trace = partial(trace, async_transport=True)
