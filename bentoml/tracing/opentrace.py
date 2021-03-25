#!/usr/bin/python
# -*- coding: utf-8 -*-


from contextlib import contextmanager
from contextvars import ContextVar
from functools import partial

import opentracing  # pylint: disable=E0401
from opentracing import Format  # pylint: disable=E0401
from opentracing.scope_managers.asyncio import (  # pylint: disable=E0401
    AsyncioScopeManager,
)
from jaeger_client.config import Config  # pylint: disable=E0401

span_context_var = ContextVar('span context', default=None)


def initialize_tracer(
    service_name, async_transport=False, host=None, port=0, sample_rate=1.0
):
    if sample_rate == 1.0:
        # sample all traces
        sampler_config = {'type': 'const', 'param': 1}
    elif sample_rate == 0.0:
        # sample none traces
        sampler_config = {'type': 'const', 'param': 0}
    else:
        # random sampling decision with the probability
        sampler_config = {'type': 'probabilistic', 'param': sample_rate}

    tracer_config = {
        'sampler': sampler_config,
    }

    if host:
        tracer_config['local_agent'] = (
            {'reporting_host': host, 'reporting_port': port},
        )

    config = Config(
        config=tracer_config,
        service_name=service_name,
        validate=True,
        scope_manager=AsyncioScopeManager() if async_transport else None,
    )

    return config.initialize_tracer()


@contextmanager
def trace(
    server_address=None,  # @UnusedVariable
    request_headers=None,
    async_transport=False,  # @UnusedVariable
    sample_rate=1.0,  # @UnusedVariable
    standalone=False,  # @UnusedVariable
    is_root=False,  # @UnusedVariable
    service_name="some service",
    span_name="service procedure",
    server_port=0,  # @UnusedVariable
):
    """
    Opentracing tracer function
    """
    tracer = initialize_tracer(
        service_name, async_transport, server_address, server_port, sample_rate
    )
    if tracer is None:
        yield
        return

    span_context = None
    span_context_saved = span_context_var.get()
    if not is_root and not standalone:
        if request_headers is not None:
            span_context = tracer.extract(Format.HTTP_HEADERS, request_headers)

        if span_context is None:
            span_context = span_context_saved or None

    with tracer.start_active_span(
        operation_name=span_name, child_of=span_context
    ) as scope:
        if standalone:
            yield None
            return
        else:
            token = span_context_var.set(scope.span.context)
            if request_headers:
                tracer.inject(
                    scope.span.context, opentracing.Format.HTTP_HEADERS, request_headers
                )
            yield scope
            span_context_var.reset(token)
            return


async_trace = partial(trace, async_transport=True)
