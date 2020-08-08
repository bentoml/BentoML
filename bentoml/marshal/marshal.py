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

import time
import asyncio
import logging
import multiprocessing
import traceback
from functools import partial

import psutil
import aiohttp

from bentoml import config
from bentoml.exceptions import RemoteException
from bentoml.server.trace import async_trace, make_http_headers
from bentoml.marshal.utils import DataLoader, SimpleRequest
from bentoml.saved_bundle import load_bento_service_metadata
from bentoml.marshal.dispatcher import CorkDispatcher, NonBlockSema
from bentoml.marshal.utils import SimpleResponse

logger = logging.getLogger(__name__)
ZIPKIN_API_URL = config("tracing").get("zipkin_api_url")


def metrics_patch(cls):
    class _MarshalService(cls):
        def __init__(self, *args, **kwargs):
            from prometheus_client import Histogram, Counter, Gauge

            super(_MarshalService, self).__init__(*args, **kwargs)
            namespace = config('instrument').get(
                'default_namespace'
            )  # its own namespace?
            service_name = self.bento_service_metadata_pb.name

            self.metrics_request_batch_size = Histogram(
                name=service_name + '_mb_batch_size',
                documentation=service_name + "microbatch request batch size",
                namespace=namespace,
                labelnames=['endpoint'],
            )
            self.metrics_request_duration = Histogram(
                name=service_name + '_mb_requestmb_duration_seconds',
                documentation=service_name + "API HTTP request duration in seconds",
                namespace=namespace,
                labelnames=['endpoint', 'http_response_code'],
            )
            self.metrics_request_in_progress = Gauge(
                name=service_name + "_mb_request_in_progress",
                documentation='Totoal number of HTTP requests in progress now',
                namespace=namespace,
                labelnames=['endpoint', 'http_method'],
            )
            self.metrics_request_exception = Counter(
                name=service_name + "_mb_request_exception",
                documentation='Totoal number of service exceptions',
                namespace=namespace,
                labelnames=['endpoint', 'exception_class'],
            )
            self.metrics_request_total = Counter(
                name=service_name + "_mb_request_total",
                documentation='Totoal number of service exceptions',
                namespace=namespace,
                labelnames=['endpoint', 'http_response_code'],
            )

        async def request_dispatcher(self, request):
            func = super(_MarshalService, self).request_dispatcher
            api_name = request.match_info.get("name", "/")
            _metrics_request_in_progress = self.metrics_request_in_progress.labels(
                endpoint=api_name, http_method=request.method,
            )
            _metrics_request_in_progress.inc()
            time_st = time.time()
            try:
                resp = await func(request)
            except Exception as e:  # pylint: disable=broad-except
                self.metrics_request_exception.labels(
                    endpoint=api_name, exception_class=e.__class__.__name__
                ).inc()
                logger.error(traceback.format_exc())
                resp = aiohttp.web.Response(status=500)
            self.metrics_request_total.labels(
                endpoint=api_name, http_response_code=resp.status
            ).inc()
            self.metrics_request_duration.labels(
                endpoint=api_name, http_response_code=resp.status
            ).observe(time.time() - time_st)
            _metrics_request_in_progress.dec()
            return resp

        async def _batch_handler_template(self, requests, api_name):
            func = super(_MarshalService, self)._batch_handler_template
            self.metrics_request_batch_size.labels(endpoint=api_name).observe(
                len(requests)
            )
            return await func(requests, api_name)

    return _MarshalService


@metrics_patch
class MarshalService:
    """
    MarshalService creates a reverse proxy server in front of actual API server,
    implementing the micro batching feature.
    It wait a short period and packed multiple requests in a single batch
    before sending to the API server.
    It applied an optimized CORK algorithm to get best efficiency.
    """

    _MARSHAL_FLAG = config("marshal_server").get("marshal_request_header_flag")
    _DEFAULT_PORT = config("apiserver").getint("default_port")
    DEFAULT_MAX_LATENCY = config("marshal_server").getint("default_max_latency")
    DEFAULT_MAX_BATCH_SIZE = config("marshal_server").getint("default_max_batch_size")

    def __init__(
        self,
        bento_bundle_path,
        outbound_host="localhost",
        outbound_port=None,
        outbound_workers=1,
    ):
        self.outbound_host = outbound_host
        self.outbound_port = outbound_port
        self.outbound_workers = outbound_workers
        self.batch_handlers = dict()
        self._outbound_sema = None  # the semaphore to limit outbound connections

        self.bento_service_metadata_pb = load_bento_service_metadata(bento_bundle_path)

        self.setup_routes_from_pb(self.bento_service_metadata_pb)
        if psutil.POSIX:
            import resource

            self.CONNECTION_LIMIT = resource.getrlimit(resource.RLIMIT_NOFILE)[0]
        else:
            self.CONNECTION_LIMIT = 1024
        logger.info(
            "Your system nofile limit is %d, which means each instance of microbatch "
            "service is able to hold this number of connections at same time. "
            "You can increase the number of file descriptors for the server process, "
            "or launch more microbatch instances to accept more concurrent connection.",
            self.CONNECTION_LIMIT,
        )

    def set_outbound_port(self, outbound_port):
        self.outbound_port = outbound_port

    def fetch_sema(self):
        if self._outbound_sema is None:
            self._outbound_sema = NonBlockSema(self.outbound_workers)
        return self._outbound_sema

    def add_batch_handler(self, api_name, max_latency, max_batch_size):
        '''
        Params:
        * max_latency: limit the max latency of overall request handling
        * max_batch_size: limit the max batch size for handler

        ** marshal server will give priority to meet these limits than efficiency
        '''

        if api_name not in self.batch_handlers:
            _func = CorkDispatcher(
                max_latency,
                max_batch_size,
                shared_sema=self.fetch_sema(),
                fallback=aiohttp.web.HTTPTooManyRequests,
            )(partial(self._batch_handler_template, api_name=api_name))
            self.batch_handlers[api_name] = _func

    def setup_routes_from_pb(self, bento_service_metadata_pb):
        from bentoml.adapters import BATCH_MODE_SUPPORTED_INPUT_TYPES

        for api_pb in bento_service_metadata_pb.apis:
            if api_pb.input_type in BATCH_MODE_SUPPORTED_INPUT_TYPES:
                max_latency = api_pb.mb_max_latency or self.DEFAULT_MAX_LATENCY
                max_batch_size = api_pb.mb_max_batch_size or self.DEFAULT_MAX_BATCH_SIZE
                self.add_batch_handler(api_pb.name, max_latency, max_batch_size)
                logger.info("Micro batch enabled for API `%s`", api_pb.name)

    async def request_dispatcher(self, request):
        with async_trace(
            ZIPKIN_API_URL,
            service_name=self.__class__.__name__,
            span_name="[1]http request",
            is_root=True,
            standalone=True,
            sample_rate=0.001,
        ):
            api_name = request.match_info.get("name")
            if api_name in self.batch_handlers:
                req = SimpleRequest(request.raw_headers, await request.read())
                try:
                    resp = await self.batch_handlers[api_name](req)
                except RemoteException as e:
                    # known remote exception
                    logger.error(traceback.format_exc())
                    resp = aiohttp.web.Response(
                        status=e.payload.status,
                        headers=e.payload.headers,
                        body=e.payload.data,
                    )
                except Exception:  # pylint: disable=broad-except
                    logger.error(traceback.format_exc())
                    resp = aiohttp.web.HTTPInternalServerError()
            else:
                resp = await self.relay_handler(request)
        return resp

    async def relay_handler(self, request):
        data = await request.read()
        headers = dict(request.headers)
        url = request.url.with_host(self.outbound_host).with_port(self.outbound_port)

        with async_trace(
            ZIPKIN_API_URL,
            service_name=self.__class__.__name__,
            span_name=f"[2]{url.path} relay",
        ) as trace_ctx:
            headers.update(make_http_headers(trace_ctx))
            async with aiohttp.ClientSession() as client:
                async with client.request(
                    request.method, url, data=data, headers=request.headers
                ) as resp:
                    body = await resp.read()
        return aiohttp.web.Response(
            status=resp.status, body=body, headers=resp.headers,
        )

    async def _batch_handler_template(self, requests, api_name):
        '''
        batch request handler
        params:
            * requests: list of aiohttp request
            * api_name: called API name
        raise:
            * RemoteException: known exceptions from model server
            * Exception: other exceptions
        '''
        headers = {self._MARSHAL_FLAG: "true"}
        api_url = f"http://{self.outbound_host}:{self.outbound_port}/{api_name}"

        with async_trace(
            ZIPKIN_API_URL,
            service_name=self.__class__.__name__,
            span_name=f"[2]merged {api_name}",
        ) as trace_ctx:
            headers.update(make_http_headers(trace_ctx))
            reqs_s = DataLoader.merge_requests(requests)
            try:
                async with aiohttp.ClientSession() as client:
                    async with client.post(
                        api_url, data=reqs_s, headers=headers
                    ) as resp:
                        raw = await resp.read()
            except aiohttp.client_exceptions.ClientConnectionError as e:
                raise RemoteException(
                    e, payload=SimpleResponse(503, None, b"Service Unavailable")
                )
            if resp.status != 200:
                raise RemoteException(
                    f"Bad response status from model server:\n{resp.status}\n{raw}",
                    payload=SimpleResponse(resp.status, resp.headers, raw),
                )
            merged = DataLoader.split_responses(raw)
            return tuple(
                aiohttp.web.Response(body=i.data, headers=i.headers, status=i.status)
                for i in merged
            )

    def async_start(self, port):
        """
        Start an micro batch server at the specific port on the instance or parameter.
        """
        marshal_proc = multiprocessing.Process(
            target=self.fork_start_app, kwargs=dict(port=port), daemon=True,
        )
        marshal_proc.start()
        logger.info("Running micro batch service on :%d", port)

    def make_app(self):
        app = aiohttp.web.Application()
        app.router.add_view("/", self.relay_handler)
        app.router.add_view("/{name}", self.request_dispatcher)
        app.router.add_view("/{path:.*}", self.relay_handler)
        return app

    def fork_start_app(self, port):
        # Use new eventloop in the fork process to avoid problems on MacOS
        # ref: https://groups.google.com/forum/#!topic/python-tornado/DkXjSNPCzsI
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        app = self.make_app()
        aiohttp.web.run_app(app, port=port)
