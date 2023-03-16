from __future__ import annotations

import json
import pickle
import typing as t
import logging
import functools
from typing import TYPE_CHECKING

from simple_di import inject
from simple_di import Provide

from bentoml.exceptions import BentoMLException
from bentoml.exceptions import ServiceUnavailable

from ..types import LazyType
from ..context import trace_context
from ..context import component_context
from ..runner.utils import Params
from ..runner.utils import PAYLOAD_META_HEADER
from ..runner.utils import payload_paramss_to_batch_params
from ..utils.metrics import metric_name
from ..utils.metrics import exponential_buckets
from ..server.base_app import BaseAppFactory
from ..runner.container import Payload
from ..runner.container import AutoContainer
from ..marshal.dispatcher import CorkDispatcher
from ..configuration.containers import BentoMLContainer

feedback_logger = logging.getLogger("bentoml.feedback")
logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from starlette.routing import BaseRoute
    from starlette.requests import Request
    from starlette.responses import Response
    from starlette.middleware import Middleware
    from opentelemetry.sdk.trace import Span

    from ..runner.runner import Runner
    from ..runner.runner import RunnerMethod


class RunnerAppFactory(BaseAppFactory):
    @inject
    def __init__(
        self,
        runner: Runner,
        worker_index: int = 0,
        enable_metrics: bool = Provide[BentoMLContainer.runners_config.metrics.enabled],
    ) -> None:
        self.runner = runner
        self.worker_index = worker_index
        self.enable_metrics = enable_metrics

        self.dispatchers: dict[str, CorkDispatcher] = {}
        for method in runner.runner_methods:
            max_batch_size = method.max_batch_size if method.config.batchable else 1
            self.dispatchers[method.name] = CorkDispatcher(
                max_latency_in_ms=method.max_latency_ms,
                max_batch_size=max_batch_size,
                fallback=functools.partial(
                    ServiceUnavailable, message="process is overloaded"
                ),
            )

    @property
    def name(self) -> str:
        return self.runner.name

    def _init_metrics_wrappers(self):
        metrics_client = BentoMLContainer.metrics_client.get()

        self.legacy_adaptive_batch_size_hist_map = {
            method.name: metrics_client.Histogram(
                name=metric_name(
                    self.runner.name,
                    self.worker_index,
                    method.name,
                    "adaptive_batch_size",
                ),
                documentation="Legacy runner adaptive batch size",
                labelnames=[],
                buckets=exponential_buckets(1, 2, method.max_batch_size),
            )
            for method in self.runner.runner_methods
        }

        max_max_batch_size = max(
            method.max_batch_size for method in self.runner.runner_methods
        )

        self.adaptive_batch_size_hist = metrics_client.Histogram(
            namespace="bentoml_runner",
            name="adaptive_batch_size",
            documentation="Runner adaptive batch size",
            labelnames=[
                "runner_name",
                "worker_index",
                "method_name",
                "service_version",
                "service_name",
            ],
            buckets=exponential_buckets(1, 2, max_max_batch_size),
        )

    @property
    def on_startup(self) -> t.List[t.Callable[[], None]]:
        on_startup = super().on_startup
        on_startup.insert(0, functools.partial(self.runner.init_local, quiet=True))
        on_startup.insert(0, self._init_metrics_wrappers)

        return on_startup

    @property
    def on_shutdown(self) -> t.List[t.Callable[[], None]]:
        on_shutdown = [self.runner.destroy]
        for dispatcher in self.dispatchers.values():
            on_shutdown.append(dispatcher.shutdown)
        on_shutdown.extend(super().on_shutdown)
        return on_shutdown

    @property
    def routes(self) -> t.List[BaseRoute]:
        """
        Setup routes for Runner server, including:

        /healthz        liveness probe endpoint
        /readyz         Readiness probe endpoint
        /metrics        Prometheus metrics endpoint

        For method in self.runner.runner_methods:
        /{method.name}  Run corresponding runnable method
        /               Run the runnable method "__call__" if presented
        """
        from starlette.routing import Route

        routes = super().routes
        for method in self.runner.runner_methods:
            path = "/" if method.name == "__call__" else "/" + method.name
            if method.config.batchable:
                routes.append(
                    Route(
                        path=path,
                        endpoint=self._mk_request_handler(runner_method=method),
                        methods=["POST"],
                    )
                )
            else:
                routes.append(
                    Route(
                        path=path,
                        endpoint=self.async_run(runner_method=method),
                        methods=["POST"],
                    )
                )
        return routes

    @property
    def middlewares(self) -> list[Middleware]:
        middlewares = super().middlewares

        # otel middleware
        import opentelemetry.instrumentation.asgi as otel_asgi  # type: ignore[import]
        from starlette.middleware import Middleware

        def client_request_hook(span: Span, _scope: t.Dict[str, t.Any]) -> None:
            if span is not None:
                span_id: int = span.context.span_id
                trace_context.request_id = span_id

        middlewares.append(
            Middleware(
                otel_asgi.OpenTelemetryMiddleware,
                excluded_urls=BentoMLContainer.tracing_excluded_urls.get(),
                default_span_details=None,
                server_request_hook=None,
                client_request_hook=client_request_hook,
                tracer_provider=BentoMLContainer.tracer_provider.get(),
            )
        )

        if self.enable_metrics:
            from .http.instruments import RunnerTrafficMetricsMiddleware

            middlewares.append(Middleware(RunnerTrafficMetricsMiddleware))

        access_log_config = BentoMLContainer.runners_config.logging.access
        if access_log_config.enabled.get():
            from .http.access import AccessLogMiddleware

            access_logger = logging.getLogger("bentoml.access")
            if access_logger.getEffectiveLevel() <= logging.INFO:
                middlewares.append(
                    Middleware(
                        AccessLogMiddleware,
                        has_request_content_length=access_log_config.request_content_length.get(),
                        has_request_content_type=access_log_config.request_content_type.get(),
                        has_response_content_length=access_log_config.response_content_length.get(),
                        has_response_content_type=access_log_config.response_content_type.get(),
                    )
                )

        return middlewares

    def _mk_request_handler(
        self,
        runner_method: RunnerMethod[t.Any, t.Any, t.Any],
    ) -> t.Callable[[Request], t.Coroutine[None, None, Response]]:
        from starlette.responses import Response

        server_str = f"BentoML-Runner/{self.runner.name}/{runner_method.name}/{self.worker_index}"

        async def infer_batch(
            params_list: t.Sequence[Params[t.Any]],
        ) -> list[Payload] | list[tuple[Payload, ...]]:
            self.legacy_adaptive_batch_size_hist_map[runner_method.name].observe(  # type: ignore
                len(params_list)
            )
            self.adaptive_batch_size_hist.labels(  # type: ignore
                runner_name=self.runner.name,
                worker_index=self.worker_index,
                method_name=runner_method.name,
                service_version=component_context.bento_version,
                service_name=component_context.bento_name,
            ).observe(len(params_list))

            if not params_list:
                return []

            input_batch_dim, output_batch_dim = runner_method.config.batch_dim

            batched_params, indices = payload_paramss_to_batch_params(
                params_list, input_batch_dim
            )

            batch_ret = await runner_method.async_run(
                *batched_params.args, **batched_params.kwargs
            )

            # multiple output branch
            if LazyType["tuple[t.Any, ...]"](tuple).isinstance(batch_ret):
                output_num = len(batch_ret)
                payloadss = tuple(
                    AutoContainer.batch_to_payloads(
                        batch_ret[idx], indices, batch_dim=output_batch_dim
                    )
                    for idx in range(output_num)
                )
                ret = list(zip(*payloadss))
                return ret

            # single output branch
            payloads = AutoContainer.batch_to_payloads(
                batch_ret,
                indices,
                batch_dim=output_batch_dim,
            )
            return payloads

        infer = self.dispatchers[runner_method.name](infer_batch)

        async def _request_handler(request: Request) -> Response:
            assert self._is_ready

            arg_num = int(request.headers["args-number"])
            r_: bytes = await request.body()

            if arg_num == 1:
                params: Params[t.Any] = _deserialize_single_param(request, r_)

            else:
                params: Params[t.Any] = pickle.loads(r_)

            try:
                payload = await infer(params)
            except BentoMLException as e:
                # pass known exceptions to the client
                return Response(
                    status_code=e.error_code,
                    content=str(e),
                    headers={
                        PAYLOAD_META_HEADER: "{}",
                        "Content-Type": "application/vnd.bentoml.error",
                        "Server": server_str,
                    },
                )
            if isinstance(payload, ServiceUnavailable):
                return Response(
                    "Service Busy",
                    status_code=payload.error_code,
                    headers={
                        PAYLOAD_META_HEADER: json.dumps({}),
                        "Content-Type": "application/vnd.bentoml.error",
                        "Server": server_str,
                    },
                )
            if isinstance(payload, Payload):
                return Response(
                    payload.data,
                    headers={
                        PAYLOAD_META_HEADER: json.dumps(payload.meta),
                        "Content-Type": f"application/vnd.bentoml.{payload.container}",
                        "Server": server_str,
                    },
                )
            if isinstance(payload, tuple):
                # a tuple, which means user runnable has multiple outputs
                return Response(
                    pickle.dumps(payload),
                    headers={
                        PAYLOAD_META_HEADER: json.dumps({}),
                        "Content-Type": "application/vnd.bentoml.multiple_outputs",
                        "Server": server_str,
                    },
                )
            raise BentoMLException(
                f"Unexpected payload type: {type(payload)}, {payload}"
            )

        return _request_handler

    def async_run(
        self,
        runner_method: RunnerMethod[t.Any, t.Any, t.Any],
    ) -> t.Callable[[Request], t.Coroutine[None, None, Response]]:
        from starlette.responses import Response

        async def _run(request: Request) -> Response:
            assert self._is_ready

            arg_num = int(request.headers["args-number"])
            r_: bytes = await request.body()

            if arg_num == 1:
                params: Params[t.Any] = _deserialize_single_param(request, r_)
            else:
                params: Params[t.Any] = pickle.loads(r_)

            params = params.map(AutoContainer.from_payload)

            try:
                ret = await runner_method.async_run(*params.args, **params.kwargs)
            except Exception as exc:  # pylint: disable=broad-except
                logger.error(
                    "Exception on runner '%s' method '%s'",
                    runner_method.runner.name,
                    runner_method.name,
                    exc_info=exc,
                )
                return Response(
                    status_code=500,
                    headers={
                        "Content-Type": "text/plain",
                        "Server": f"BentoML-Runner/{self.runner.name}/{runner_method.name}/{self.worker_index}",
                    },
                )
            else:
                payload = AutoContainer.to_payload(ret, 0)
                return Response(
                    payload.data,
                    headers={
                        PAYLOAD_META_HEADER: json.dumps(payload.meta),
                        "Content-Type": f"application/vnd.bentoml.{payload.container}",
                        "Server": f"BentoML-Runner/{self.runner.name}/{runner_method.name}/{self.worker_index}",
                    },
                )

        return _run


def _deserialize_single_param(request: Request, bs: bytes) -> Params[t.Any]:

    container = request.headers["Payload-Container"]
    meta = json.loads(request.headers["Payload-Meta"])
    batch_size = int(request.headers["Batch-Size"])
    kwarg_name = request.headers.get("Kwarg-Name")
    payload = Payload(
        data=bs,
        meta=meta,
        batch_size=batch_size,
        container=container,
    )
    if kwarg_name:
        d = {kwarg_name: payload}
        params: Params[t.Any] = Params(**d)
    else:
        params: Params[t.Any] = Params(payload)

    return params
