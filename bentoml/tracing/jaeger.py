# Copyright 2021 Atalaya Tech, Inc.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

# http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from contextlib import contextmanager
from contextvars import ContextVar

span_context_var = ContextVar('span context', default=None)


def initialize_tracer(
    service_name, async_transport=False, host=None, port=0, sample_rate=1.0
):
    from opentracing.scope_managers.asyncio import (  # pylint: disable=E0401
        AsyncioScopeManager,
    )
    from jaeger_client.config import Config  # pylint: disable=E0401

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
        tracer_config['local_agent'] = {'reporting_host': host, 'reporting_port': port}

    config = Config(
        config=tracer_config,
        service_name=service_name,
        validate=True,
        scope_manager=AsyncioScopeManager() if async_transport else None,
    )

    return config.new_tracer()


class JaegerTracer:
    def __init__(self, address, port):
        self.address = address
        self.port = port

    @contextmanager
    def span(
        self,
        service_name,
        span_name,
        request_headers=None,
        async_transport=False,
        sample_rate=1.0,
        standalone=False,
        is_root=False,
    ):
        """
        Opentracing tracer function
        """
        from opentracing import Format  # pylint: disable=E0401
        from jaeger_client.constants import TRACE_ID_HEADER  # pylint: disable=E0401

        tracer = initialize_tracer(
            service_name, async_transport, self.address, self.port, sample_rate
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
                if request_headers and TRACE_ID_HEADER not in request_headers:
                    tracer.inject(
                        scope.span.context, Format.HTTP_HEADERS, request_headers,
                    )
                yield scope
                span_context_var.reset(token)
                return

    @contextmanager
    def async_span(self, *args, **kwargs):
        with self.span(*args, async_transport=True, **kwargs) as ctx:
            yield ctx
        return


def get_jaeger_tracer(address, port):
    return JaegerTracer(address, port)
