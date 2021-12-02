import time
import asyncio
import logging
import functools
import traceback
from typing import Optional
from typing import TYPE_CHECKING

import psutil
from simple_di import inject
from simple_di import Provide

from .utils import DataLoader
from .utils import MARSHAL_REQUEST_HEADER
from ..types import HTTPRequest
from ..types import HTTPResponse
from .dispatcher import NonBlockSema
from .dispatcher import CorkDispatcher
from ...exceptions import RemoteException

# from ..bundle import load_bento_service_metadata
from ..bundle.config import DEFAULT_MAX_LATENCY
from ..bundle.config import DEFAULT_MAX_BATCH_SIZE
from ..configuration.containers import BentoMLContainer

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from aiohttp import BaseConnector
    from aiohttp import ClientSession
    from aiohttp.web import Request
    from aiohttp.web import Application


def metrics_patch(cls):
    class _MarshalApp(cls):
        @inject
        def __init__(
            self,
            *args,
            metrics_client=Provide[BentoMLContainer.metrics_client],
            **kwargs,
        ):
            for attr_name in functools.WRAPPER_ASSIGNMENTS:
                try:
                    setattr(self.__class__, attr_name, getattr(cls, attr_name))
                except AttributeError:
                    pass

            super(_MarshalApp, self).__init__(*args, **kwargs)
            # its own namespace?
            service_name = self.bento_service_metadata_pb.name

            self.metrics_request_batch_size = metrics_client.Histogram(
                name=service_name + "_mb_batch_size",
                documentation=service_name + "microbatch request batch size",
                labelnames=["endpoint"],
            )
            self.metrics_request_duration = metrics_client.Histogram(
                name=service_name + "_mb_request_duration_seconds",
                documentation=service_name + "API HTTP request duration in seconds",
                labelnames=["endpoint", "http_response_code"],
            )
            self.metrics_request_in_progress = metrics_client.Gauge(
                name=service_name + "_mb_request_in_progress",
                documentation="Total number of HTTP requests in progress now",
                labelnames=["endpoint", "http_method"],
            )
            self.metrics_request_exception = metrics_client.Counter(
                name=service_name + "_mb_request_exception",
                documentation="Total number of service exceptions",
                labelnames=["endpoint", "exception_class"],
            )
            self.metrics_request_total = metrics_client.Counter(
                name=service_name + "_mb_request_total",
                documentation="Total number of service exceptions",
                labelnames=["endpoint", "http_response_code"],
            )

        async def request_dispatcher(self, request):
            from aiohttp.web import Response

            func = super(_MarshalApp, self).request_dispatcher
            api_route = request.match_info.get("path", "/")
            _metrics_request_in_progress = self.metrics_request_in_progress.labels(
                endpoint=api_route,
                http_method=request.method,
            )
            _metrics_request_in_progress.inc()
            time_st = time.time()
            try:
                resp = await func(request)
            except asyncio.CancelledError:
                resp = Response(status=503)
            except Exception as e:  # pylint: disable=broad-except
                self.metrics_request_exception.labels(
                    endpoint=api_route, exception_class=e.__class__.__name__
                ).inc()
                logger.error(traceback.format_exc())
                resp = Response(status=500)
            self.metrics_request_total.labels(
                endpoint=api_route, http_response_code=resp.status
            ).inc()
            self.metrics_request_duration.labels(
                endpoint=api_route, http_response_code=resp.status
            ).observe(time.time() - time_st)
            _metrics_request_in_progress.dec()
            return resp

        async def _batch_handler_template(self, requests, api_route, max_latency):
            func = super(_MarshalApp, self)._batch_handler_template
            self.metrics_request_batch_size.labels(endpoint=api_route).observe(
                len(requests)
            )
            return await func(requests, api_route, max_latency)

    return _MarshalApp


@metrics_patch
class MarshalApp:
    """
    MarshalApp creates a reverse proxy server in front of actual API server,
    implementing the micro batching feature.
    It wait a short period and packed multiple requests in a single batch
    before sending to the API server.
    It applied an optimized CORK algorithm to get best efficiency.
    """

    @inject
    def __init__(
        self,
        bento_bundle_path: str = Provide[BentoMLContainer.bundle_path],
        outbound_host: str = Provide[BentoMLContainer.forward_host],
        outbound_port: int = Provide[BentoMLContainer.forward_port],
        outbound_workers: int = Provide[BentoMLContainer.api_server_workers],
        mb_max_batch_size: int = Provide[
            BentoMLContainer.config.bento_server.microbatch.max_batch_size
        ],
        mb_max_latency: int = Provide[
            BentoMLContainer.config.bento_server.microbatch.max_latency
        ],
        max_request_size: int = Provide[
            BentoMLContainer.config.bento_server.max_request_size
        ],
        outbound_unix_socket: str = None,
        timeout: int = Provide[BentoMLContainer.config.bento_server.timeout],
        tracer=Provide[BentoMLContainer.tracer],
    ):

        self._conn: Optional["BaseConnector"] = None
        self._client: Optional["ClientSession"] = None
        self.outbound_unix_socket = outbound_unix_socket
        self.outbound_host = outbound_host
        self.outbound_port = outbound_port
        self.outbound_workers = outbound_workers
        self.mb_max_batch_size = mb_max_batch_size
        self.mb_max_latency = mb_max_latency
        self.batch_handlers = dict()
        self._outbound_sema = None  # the semaphore to limit outbound connections
        self._cleanup_tasks = None
        self.max_request_size = max_request_size
        self.tracer = tracer

        # self.bento_service_metadata_pb = load_bento_service_metadata(bento_bundle_path)

        self.setup_routes_from_pb(self.bento_service_metadata_pb)
        self.timeout = timeout

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

    @property
    def cleanup_tasks(self):
        if self._cleanup_tasks is None:
            self._cleanup_tasks = []
        return self._cleanup_tasks

    async def cleanup(self, _):
        # clean up futures for gracefully shutting down
        for task in self.cleanup_tasks:
            await task()

        if hasattr(self, "_client"):
            if self._client is not None and not self._client.closed:
                await self._client.close()

    def fetch_sema(self):
        if self._outbound_sema is None:
            self._outbound_sema = NonBlockSema(self.outbound_workers)
        return self._outbound_sema

    def get_conn(self) -> "BaseConnector":
        import aiohttp

        if self._conn is None or self._conn.closed:
            if self.outbound_unix_socket:
                self._conn = aiohttp.UnixConnector(
                    path=self.outbound_unix_socket,
                )
            else:
                self._conn = aiohttp.TCPConnector(limit=30)
        return self._conn

    def get_client(self):
        import aiohttp

        if self._client is None or self._client.closed:
            jar = aiohttp.DummyCookieJar()
            if self.timeout:
                timeout = aiohttp.ClientTimeout(total=self.timeout)
            else:
                timeout = None
            self._client = aiohttp.ClientSession(
                connector=self.get_conn(),
                auto_decompress=False,
                cookie_jar=jar,
                connector_owner=False,
                timeout=timeout,
            )
        return self._client

    def add_batch_handler(self, api_route, max_latency, max_batch_size):
        """
        Params:
        * max_latency: limit the max latency of overall request handling
        * max_batch_size: limit the max batch size for handler

        ** marshal server will give priority to meet these limits than efficiency
        """
        from aiohttp.web import HTTPTooManyRequests

        if api_route not in self.batch_handlers:
            dispatcher = CorkDispatcher(
                max_latency,
                max_batch_size,
                shared_sema=self.fetch_sema(),
                fallback=HTTPTooManyRequests,
            )
            _func = dispatcher(
                functools.partial(
                    self._batch_handler_template,
                    api_route=api_route,
                    max_latency=max_latency,
                )
            )
            self.cleanup_tasks.append(dispatcher.shutdown)
            self.batch_handlers[api_route] = _func

    def setup_routes_from_pb(self, bento_service_metadata_pb):
        for api_pb in bento_service_metadata_pb.apis:
            if api_pb.batch:
                max_latency = (
                    self.mb_max_latency or api_pb.mb_max_latency or DEFAULT_MAX_LATENCY
                )
                max_batch_size = (
                    self.mb_max_batch_size
                    or api_pb.mb_max_batch_size
                    or DEFAULT_MAX_BATCH_SIZE
                )
                self.add_batch_handler(api_pb.route, max_latency, max_batch_size)
                logger.info(
                    "Micro batch enabled for API `%s` max-latency: %s"
                    " max-batch-size %s",
                    api_pb.route,
                    max_latency,
                    max_batch_size,
                )

    async def request_dispatcher(self, request: "Request"):
        from aiohttp.web import Response
        from aiohttp.web import HTTPInternalServerError

        with self.tracer.async_span(
            service_name=self.__class__.__name__,
            span_name="[1]http request",
            is_root=True,
            standalone=True,
            sample_rate=0.001,
        ):
            api_route = request.match_info.get("path")
            if api_route in self.batch_handlers:
                req = HTTPRequest(
                    tuple((k.decode(), v.decode()) for k, v in request.raw_headers),
                    await request.read(),
                )
                try:
                    resp = await self.batch_handlers[api_route](req)
                except RemoteException as e:
                    # known remote exception
                    logger.error(traceback.format_exc())
                    resp = Response(
                        status=e.payload.status,
                        headers=e.payload.headers,
                        body=e.payload.body,
                    )
                except Exception:  # pylint: disable=broad-except
                    logger.error(traceback.format_exc())
                    resp = HTTPInternalServerError()
            else:
                resp = await self.relay_handler(request)
        return resp

    async def relay_handler(self, request: "Request"):
        from aiohttp.web import Response
        from aiohttp.client_exceptions import ClientConnectionError

        data = await request.read()
        url = request.url.with_host(self.outbound_host).with_port(self.outbound_port)

        with self.tracer.async_span(
            service_name=self.__class__.__name__,
            span_name=f"[2]{url.path} relay",
            request_headers=request.headers,
        ):
            try:
                client = self.get_client()
                async with client.request(
                    request.method, url, data=data, headers=request.headers
                ) as resp:
                    body = await resp.read()
            except ClientConnectionError:
                return Response(status=503, body=b"Service Unavailable")
        return Response(
            status=resp.status,
            body=body,
            headers=resp.headers,
        )

    async def _batch_handler_template(self, requests, api_route, max_latency):
        """
        batch request handler
        params:
            * requests: list of aiohttp request
            * api_route: called API name
        raise:
            * RemoteException: known exceptions from model server
            * Exception: other exceptions
        """
        from aiohttp import ClientTimeout
        from aiohttp.web import Response
        from aiohttp.client_exceptions import ClientConnectionError

        headers = {MARSHAL_REQUEST_HEADER: "true"}
        api_url = f"http://{self.outbound_host}:{self.outbound_port}/{api_route}"

        with self.tracer.async_span(
            service_name=self.__class__.__name__,
            span_name=f"[2]merged {api_route}",
            request_headers=headers,
        ):
            reqs_s = DataLoader.merge_requests(requests)
            try:
                client = self.get_client()
                timeout = ClientTimeout(
                    total=(self.mb_max_latency or max_latency) // 1000
                )
                async with client.post(
                    api_url, data=reqs_s, headers=headers, timeout=timeout
                ) as resp:
                    raw = await resp.read()
            except ClientConnectionError as e:
                raise RemoteException(
                    repr(e),
                    payload=HTTPResponse.new(status=503, body=b"Service Unavailable"),
                )
            except asyncio.CancelledError as e:
                raise RemoteException(
                    repr(e),
                    payload=HTTPResponse(
                        status=500, body=b"Cancelled before upstream responses"
                    ),
                )
            except asyncio.TimeoutError as e:
                raise RemoteException(
                    repr(e),
                    payload=HTTPResponse(status=408, body=b"Request timeout"),
                )

            if resp.status != 200:
                raise RemoteException(
                    f"Bad response status from model server:\n{resp.status}\n{raw}",
                    payload=HTTPResponse.new(
                        status=resp.status,
                        headers=tuple(resp.headers.items()),
                        body=raw,
                    ),
                )
            merged = DataLoader.split_responses(raw)
            return tuple(
                Response(body=i.body, headers=i.headers, status=i.status or 500)
                for i in merged
            )

    def get_app(self) -> "Application":
        from starlette.applications import Starlette

        app = Starlette()

        # app = Application(client_max_size=self.max_request_size)
        # app.on_cleanup.append(self.cleanup)

        ALL_METHODS = ["DELETE", "GET", "HEAD", "OPTIONS", "PATCH", "POST", "PUT"]

        app.add_route(methods=ALL_METHODS, path="/", route=self.relay_handler)
        # app.router.add_route(method, "/{path:.*}", self.request_dispatcher)

        return app

    @inject
    def run(
        self,
        port=Provide[BentoMLContainer.config.bento_server.port],
    ):
        logger.info("Starting BentoML API proxy in development mode..")
        from aiohttp.web import run_app

        # Use new eventloop in the fork process to avoid problems on MacOS
        # ref: https://groups.google.com/forum/#!topic/python-tornado/DkXjSNPCzsI
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        app = self.get_app()
        run_app(app, port=port)
