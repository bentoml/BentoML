# Copyright 2019 Atalaya Tech, Inc.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

# http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import random
import aiohttp
import asyncio
from contextlib import contextmanager
from contextvars import ContextVar
from functools import partial

from py_zipkin.zipkin import ZipkinAttrs, zipkin_span
from py_zipkin.transport import SimpleHTTPTransport
from py_zipkin.util import generate_random_64bit_string


trace_stack_var = ContextVar('trace_stack', default=None)


def load_http_headers(headers):
    if not headers or "X-B3-TraceId" not in headers:
        return None

    return ZipkinAttrs(
        headers.get("X-B3-TraceId"),
        headers.get("X-B3-SpanId"),
        headers.get("X-B3-ParentSpanId"),
        headers.get("X-B3-Flags") or '0',
        False if headers.get("X-B3-Sampled") == '0' else True,
    )


def _set_http_headers(attrs, headers):
    if not headers or "X-B3-TraceId" in headers:
        return

    tracing_headers = {
        "X-B3-TraceId": attrs.trace_id,
        "X-B3-SpanId": attrs.span_id,
        "X-B3-Flags": attrs.flags,
        "X-B3-Sampled": attrs.is_sampled and '1' or '0',
    }
    if attrs.parent_span_id:
        tracing_headers["X-B3-ParentSpanId"] = attrs.parent_span_id
    headers.update(tracing_headers)


def _make_child_attrs(attrs):
    return ZipkinAttrs(
        attrs.trace_id,
        generate_random_64bit_string(),
        attrs.span_id,
        attrs.flags,
        attrs.is_sampled,
    )


def _make_new_attrs(sample_rate=1.0):
    return ZipkinAttrs(
        generate_random_64bit_string(),
        generate_random_64bit_string(),
        None,
        '0',
        sample_rate and random.random() < sample_rate or False,
    )


class AsyncHttpTransport(SimpleHTTPTransport):
    '''
    add trace data transporting task into default eventloop
    '''

    @staticmethod
    async def _async_post(url, data, headers):
        async with aiohttp.ClientSession() as client:
            async with client.post(url, data=data, headers=headers) as resp:
                resp = await resp.text()
                return resp

    def send(self, payload):
        path, content_type = self._get_path_content_type(payload)
        url = 'http://{}:{}{}'.format(self.address, self.port, path)

        asyncio.get_event_loop().create_task(
            self._async_post(url, data=payload, headers={'Content-Type': content_type},)
        )


@contextmanager
def trace(
    server_address,
    request_headers=None,
    async_transport=False,
    sample_rate=1.0,
    standalone=False,
    is_root=False,
    service_name="some service",
    span_name="service procedure",
    server_port=0,
):
    trace_stack = trace_stack_var.get()

    parent_attrs = load_http_headers(request_headers) or trace_stack or None

    if not is_root and parent_attrs:
        attrs = _make_child_attrs(parent_attrs)
    else:
        attrs = _make_new_attrs(sample_rate)

    if not attrs.is_sampled or not server_address:
        if standalone:
            yield None
            return
        else:
            token = trace_stack_var.set(attrs)
            _set_http_headers(attrs, request_headers)
            yield attrs
            trace_stack_var.reset(token)
            return

    if async_transport:
        transport_handler = AsyncHttpTransport(server_address, server_port)
    else:
        transport_handler = SimpleHTTPTransport(server_address, server_port)

    with zipkin_span(
        service_name=service_name,
        span_name=span_name,
        zipkin_attrs=attrs,
        transport_handler=transport_handler,
    ):
        if standalone:
            yield None
            return
        else:
            token = trace_stack_var.set(attrs)
            _set_http_headers(attrs, request_headers)
            yield
            trace_stack_var.reset(token)


async_trace = partial(trace, async_transport=True)
