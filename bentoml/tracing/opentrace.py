#!/usr/bin/python
# -*- coding: utf-8 -*-

from contextlib import contextmanager
from contextvars import ContextVar
from functools import partial

# pylint: disable=E0401
import opentracing
from opentracing import Format
from jaeger_client.config import Config
from opentracing.scope_managers.asyncio import AsyncioScopeManager

# pylint: enable=E0401


span_context_var = ContextVar('span context', default=None)


def initialize_tracer(service_name):
    config = Config(
        config={'sampler': {'type': 'const', 'param': 1}},
        service_name=service_name,
        validate=True,
        scope_manager=AsyncioScopeManager(),
    )

    return config.initialize_tracer()


@contextmanager
def trace(
    server_url=None,  # @UnusedVariable
    request_headers=None,
    async_transport=False,  # @UnusedVariable
    sample_rate=1.0,  # @UnusedVariable
    standalone=False,  # @UnusedVariable
    is_root=False,  # @UnusedVariable
    service_name="some service",
    span_name="service procedure",
    port=0,  # @UnusedVariable
):
    """
    Opentracing tracer function
    """
    del server_url, async_transport, sample_rate, standalone, is_root, port

    tracer = initialize_tracer(service_name) or opentracing.global_tracer() or None
    if tracer is None:
        yield
        return

    span_context = None
    span_context_saved = span_context_var.get()

    if request_headers is not None:
        span_context = tracer.extract(Format.HTTP_HEADERS, request_headers)

    if span_context is None:
        span_context = span_context_saved or None

    with tracer.start_active_span(
        operation_name=span_name, child_of=span_context
    ) as scope:
        token = span_context_var.set(scope.span.context)
        yield scope
        span_context_var.reset(token)


async_trace = partial(trace, async_transport=True)
