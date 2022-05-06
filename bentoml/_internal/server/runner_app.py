from __future__ import annotations

import json
import typing as t
import asyncio
import logging
from typing import TYPE_CHECKING
from functools import partial

from ..trace import ServiceContext
from ..runner.utils import PAYLOAD_META_HEADER
from ..runner.utils import multipart_to_payload_params
from ..server.base_app import BaseAppFactory
from ..runner.container import Payload
from ..runner.container import AutoContainer
from ..marshal.dispatcher import CorkDispatcher
from ..configuration.containers import DeploymentContainer

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


T = t.TypeVar("T")


class RunnerAppFactory(BaseAppFactory):
    def __init__(
        self,
        runner: Runner,
        instance_id: int | None = None,
    ) -> None:
        self.runner = runner
        self.instance_id = instance_id

        from starlette.responses import Response

        TooManyRequests = partial(Response, status_code=429)

        self.dispatchers: dict[str, CorkDispatcher] = {}
        for method in runner.runner_methods:
            if not method.runnable_method_config.batchable:
                continue
            self.dispatchers[method.name] = CorkDispatcher(
                max_latency_in_ms=method.max_latency_ms,
                max_batch_size=method.max_batch_size,
                fallback=TooManyRequests,
            )

    @property
    def name(self) -> str:
        return self.runner.name

    @property
    def on_startup(self) -> t.List[t.Callable[[], None]]:
        on_startup = super().on_startup
        on_startup.insert(0, self.runner._init_local)  # type: ignore
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

        /run
        /run_batch
        """
        from starlette.routing import Route

        routes = super().routes
        for method in self.runner.runner_methods:
            path = "/" if method.name == "__call__" else "/" + method.name
            if method.runnable_method_config.batchable:
                _func = self.dispatchers[method.name](
                    self._async_cork_run(runner_method=method)
                )
                routes.append(
                    Route(
                        path=path,
                        endpoint=_func,
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
                ServiceContext.request_id_var.set(span_id)

        def client_response_hook(span: Span, _message: t.Any) -> None:
            if span is not None:
                ServiceContext.request_id_var.set(None)

        middlewares.append(
            Middleware(
                otel_asgi.OpenTelemetryMiddleware,
                excluded_urls=None,
                default_span_details=None,
                server_request_hook=None,
                client_request_hook=client_request_hook,
                client_response_hook=client_response_hook,
                tracer_provider=DeploymentContainer.tracer_provider.get(),
            )
        )

        access_log_config = DeploymentContainer.runners_config.logging.access
        if access_log_config.enabled.get():
            from .access import AccessLogMiddleware

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

    def _async_cork_run(
        self,
        runner_method: RunnerMethod,
    ) -> t.Callable[[t.Iterable[Request]], t.Coroutine[None, None, list[Response]]]:
        from starlette.responses import Response

        async def _run(requests: t.Iterable[Request]) -> list[Response]:
            assert self._is_ready

            params_list = await asyncio.gather(
                *tuple(multipart_to_payload_params(r) for r in requests)
            )

            batch_dim = runner_method.runnable_method_config.batch_dim

            indices: list[int] = []

            i = 0
            batch_args: list[t.Any] = []
            while i in batch_dim.args:
                args = [param.args[i] for param in params_list]
                batched_arg, indices = AutoContainer.from_batch_payloads(
                    args, batch_dim=batch_dim[i]
                )
                batch_args.append(batched_arg)
                i += 1

            found = True
            while found:
                found = False
                args: list[t.Any] = []
                for params in params_list:
                    if len(params.args) > i:
                        args.append(None)
                        continue
                    else:
                        found = True
                        args.append(params.args[i])
                if found:
                    batched_arg, indices = AutoContainer.from_batch_payloads(
                        args, batch_dim=batch_dim[i]
                    )
                    batch_args.append(batched_arg)
                    i += 1

            kwargs: dict[str, list[Payload | None]] = {}
            for i, params in enumerate(params_list):
                for kwarg in params.kwargs:
                    kwarg_list: list[Payload | None] = kwargs.get(kwarg, [None] * i)
                    kwarg_list.append(params.kwargs[kwarg])
                    kwargs[kwarg] = kwarg_list

            batch_kwargs = {}
            for kwarg, payloads in kwargs.items():
                batch_kwargs[kwargs], indices = AutoContainer.from_batch_payloads(
                    payloads, batch_dim=batch_dim[kwarg]
                )

            batch_ret = await runner_method.async_run(*batch_args, **batch_kwargs)

            payloads = AutoContainer.batch_to_payloads(
                batch_ret,
                indices,
                batch_dim=batch_dim[-1],
            )
            return [
                Response(
                    payload.data,
                    headers={
                        PAYLOAD_META_HEADER: json.dumps(payload.meta),
                        "Content-Type": f"application/vnd.bentoml.{payload.container}",
                        "Server": f"BentoML-Runner/{self.runner.name}/{runner_method.name}/{self.instance_id}",
                    },
                )
                for payload in payloads
            ]

        return _run

    def async_run(
        self,
        runner_method: RunnerMethod,
    ) -> t.Callable[[Request], t.Coroutine[None, None, Response]]:
        from starlette.responses import Response

        async def _run(request: Request) -> Response:
            assert self._is_ready

            logger.info(request)
            params = await multipart_to_payload_params(request)
            params = params.map(AutoContainer.from_payload)
            ret = await runner_method.async_run(*params.args, **params.kwargs)
            payload = AutoContainer.to_payload(ret)
            return Response(
                payload.data,
                headers={
                    PAYLOAD_META_HEADER: json.dumps(payload.meta),
                    "Content-Type": f"application/vnd.bentoml.{payload.container}",
                    "Server": f"BentoML-Runner/{self.runner.name}/{runner_method.name}/{self.instance_id}",
                },
            )

        return _run
