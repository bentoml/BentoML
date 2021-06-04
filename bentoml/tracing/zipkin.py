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
import asyncio
import requests
from contextlib import contextmanager
from contextvars import ContextVar


trace_stack_var = ContextVar('trace_stack', default=None)


def _load_http_headers(headers):
    from py_zipkin.zipkin import ZipkinAttrs  # pylint: disable=E0401

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
    from py_zipkin.zipkin import ZipkinAttrs  # pylint: disable=E0401
    from py_zipkin.util import generate_random_64bit_string  # pylint: disable=E0401

    return ZipkinAttrs(
        attrs.trace_id,
        generate_random_64bit_string(),
        attrs.span_id,
        attrs.flags,
        attrs.is_sampled,
    )


def _make_new_attrs(sample_rate=1.0):
    from py_zipkin.zipkin import ZipkinAttrs  # pylint: disable=E0401
    from py_zipkin.util import generate_random_64bit_string  # pylint: disable=E0401

    return ZipkinAttrs(
        generate_random_64bit_string(),
        generate_random_64bit_string(),
        None,
        '0',
        sample_rate and random.random() < sample_rate or False,
    )


def get_zipkin_tracer(server_url):
    from py_zipkin.transport import BaseTransportHandler  # pylint: disable=E0401

    class HttpTransport(BaseTransportHandler):
        def __init__(self, server_url):
            super(HttpTransport, self).__init__()
            self.server_url = server_url

        def get_max_payload_bytes(self):
            # None for no max payload size
            return None

        def send(self, payload):
            requests.post(
                self.server_url,
                data=payload,
                headers={'Content-Type': 'application/x-thrift'},
            )

    class AsyncHttpTransport(BaseTransportHandler):
        '''
        add trace data transporting task into default eventloop
        '''

        def __init__(self, server_url):
            super(AsyncHttpTransport, self).__init__()
            self.server_url = server_url

        def get_max_payload_bytes(self):
            # None for no max payload size
            return None

        @staticmethod
        async def _async_post(url, data, headers):
            from aiohttp import ClientSession

            async with ClientSession() as client:
                async with client.post(url, data=data, headers=headers) as resp:
                    resp = await resp.text()
                    return resp

        def send(self, payload):
            asyncio.get_event_loop().create_task(
                self._async_post(
                    self.server_url,
                    data=payload,
                    headers={'Content-Type': 'application/x-thrift'},
                )
            )

    class ZipkinTracer:
        def __init__(self, server_url):
            self.server_url = server_url
            self.async_transport = AsyncHttpTransport(self.server_url)
            self.http_transport = HttpTransport(self.server_url)

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
            from py_zipkin import Encoding  # pylint: disable=E0401
            from py_zipkin.zipkin import zipkin_span  # pylint: disable=E0401

            trace_stack = trace_stack_var.get()

            parent_attrs = _load_http_headers(request_headers) or trace_stack or None

            if not is_root and parent_attrs:
                attrs = _make_child_attrs(parent_attrs)
            else:
                attrs = _make_new_attrs(sample_rate)

            if not attrs.is_sampled:
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
                transport_handler = self.async_transport
            else:
                transport_handler = self.http_transport

            with zipkin_span(
                service_name=service_name,
                span_name=span_name,
                zipkin_attrs=attrs,
                transport_handler=transport_handler,
                encoding=Encoding.V2_JSON,
            ):
                if standalone:
                    yield
                    return
                else:
                    token = trace_stack_var.set(attrs)
                    _set_http_headers(attrs, request_headers)
                    yield
                    trace_stack_var.reset(token)

        @contextmanager
        def async_span(self, *args, **kwargs):
            with self.span(*args, async_transport=True, **kwargs) as ctx:
                yield ctx
            return

    return ZipkinTracer(server_url)
