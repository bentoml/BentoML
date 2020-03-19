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
import resource
import asyncio
import logging
import multiprocessing
from functools import partial

import aiohttp
from bentoml import config
from bentoml.utils.trace import async_trace, make_http_headers
from bentoml.marshal.utils import DataLoader, SimpleRequest
from bentoml.handlers import HANDLER_TYPES_BATCH_MODE_SUPPORTED
from bentoml.bundler import load_bento_service_metadata
from bentoml.utils.usage_stats import track_server
from bentoml.marshal.dispatcher import ParadeDispatcher

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
            api_name = request.match_info["name"]
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
    Requests in a short period(mb_max_latency) are collected and sent to API server,
    merged into a single request.
    """

    _MARSHAL_FLAG = config("marshal_server").get("marshal_request_header_flag")
    _DEFAULT_PORT = config("apiserver").getint("default_port")
    _DEFAULT_MAX_LATENCY = config("marshal_server").getint("default_max_latency")
    _DEFAULT_MAX_BATCH_SIZE = config("marshal_server").getint("default_max_batch_size")

    def __init__(
        self,
        bento_bundle_path,
        target_host="localhost",
        target_port=None,
        target_count=2,
    ):
        self.target_host = target_host
        self.target_port = target_port
        self.target_count = target_count
        self.batch_handlers = dict()
        self._target_sema = None

        self.bento_service_metadata_pb = load_bento_service_metadata(bento_bundle_path)

        self.setup_routes_from_pb(self.bento_service_metadata_pb)

        self.CONNECTION_LIMIT = resource.getrlimit(resource.RLIMIT_NOFILE)[0]
        logger.info(
            "Your system nofile limit is %d, which means each instance of microbatch "
            "service is able to hold this number of connections at same time. "
            "You can increase the number of file descriptors for the server process, "
            "or launch more microbatch instances to accept more concurrent connection.",
            self.CONNECTION_LIMIT,
        )

    def set_target_port(self, target_port):
        self.target_port = target_port

    def fetch_sema(self):
        if self._target_sema is None:
            self._target_sema = asyncio.Semaphore(self.target_count * 2)
        return self._target_sema

    def add_batch_handler(self, api_name, max_latency, max_batch_size):

        if api_name not in self.batch_handlers:
            _func = ParadeDispatcher(
                max_latency, max_batch_size, shared_sema=self.fetch_sema
            )(partial(self._batch_handler_template, api_name=api_name))
            self.batch_handlers[api_name] = _func

    def setup_routes_from_pb(self, bento_service_metadata_pb):
        for api_config in bento_service_metadata_pb.apis:
            if api_config.handler_type in HANDLER_TYPES_BATCH_MODE_SUPPORTED:
                handler_config = getattr(api_config, "handler_config", {})
                max_latency = (
                    handler_config["mb_max_latency"]
                    if "mb_max_latency" in handler_config
                    else self._DEFAULT_MAX_LATENCY
                )
                self.add_batch_handler(
                    api_config.name, max_latency, self._DEFAULT_MAX_BATCH_SIZE
                )
                logger.info("Micro batch enabled for API `%s`", api_config.name)

    async def request_dispatcher(self, request):
        with async_trace(
            ZIPKIN_API_URL,
            service_name=self.__class__.__name__,
            span_name=f"[1]http request",
            is_root=True,
            standalone=True,
            sample_rate=0.001,
        ):
            api_name = request.match_info["name"]
            if api_name in self.batch_handlers:
                req = SimpleRequest(request.raw_headers, await request.read())
                resp = await self.batch_handlers[api_name](req)
            else:
                resp = await self._relay_handler(request, api_name)
        return resp

    async def _relay_handler(self, request, api_name):
        data = await request.read()
        headers = dict(request.headers)
        api_url = f"http://{self.target_host}:{self.target_port}/{api_name}"

        with async_trace(
            ZIPKIN_API_URL,
            service_name=self.__class__.__name__,
            span_name=f"[2]{api_name} relay",
        ) as trace_ctx:
            headers.update(make_http_headers(trace_ctx))
            async with aiohttp.ClientSession() as client:
                async with client.request(
                    request.method, api_url, data=data, headers=request.headers
                ) as resp:
                    body = await resp.read()
        return aiohttp.web.Response(
            status=resp.status, body=body, headers=resp.headers,
        )

    async def _batch_handler_template(self, requests, api_name):
        headers = {self._MARSHAL_FLAG: "true"}
        api_url = f"http://{self.target_host}:{self.target_port}/{api_name}"

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
                merged = DataLoader.split_responses(raw)
            except (aiohttp.ClientConnectorError, aiohttp.ServerDisconnectedError):
                return (aiohttp.web.HTTPServiceUnavailable,) * len(requests)

        if merged is None:
            return (aiohttp.web.HTTPInternalServerError,) * len(requests)
        return tuple(
            aiohttp.web.Response(body=i.data, headers=i.headers, status=i.status)
            for i in merged
        )

    def async_start(self, port):
        """
        Start an micro batch server at the specific port on the instance or parameter.
        """
        track_server('marshal')
        marshal_proc = multiprocessing.Process(
            target=self.fork_start_app, kwargs=dict(port=port), daemon=True,
        )
        # TODO: make sure child process dies when parent process is killed.
        marshal_proc.start()
        logger.info("Running micro batch service on :%d", port)

    def make_app(self):
        app = aiohttp.web.Application()
        app.router.add_view("/{name}", self.request_dispatcher)
        return app

    def fork_start_app(self, port):
        # Use new eventloop in the fork process to avoid problems on MacOS
        # ref: https://groups.google.com/forum/#!topic/python-tornado/DkXjSNPCzsI
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        app = self.make_app()
        aiohttp.web.run_app(app, port=port)
