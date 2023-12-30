from __future__ import annotations

import asyncio
import functools
import inspect
import logging
import sys
import typing as t
from pathlib import Path

import pydantic
from simple_di import Provide
from simple_di import inject
from starlette.middleware import Middleware
from starlette.staticfiles import StaticFiles

from _bentoml_sdk import Service
from bentoml._internal.container import BentoMLContainer
from bentoml._internal.marshal.dispatcher import CorkDispatcher
from bentoml._internal.server.base_app import BaseAppFactory
from bentoml._internal.server.http_app import log_exception
from bentoml.exceptions import BentoMLException
from bentoml.exceptions import ServiceUnavailable

if t.TYPE_CHECKING:
    from opentelemetry.sdk.trace import Span
    from starlette.applications import Starlette
    from starlette.requests import Request
    from starlette.responses import Response
    from starlette.routing import BaseRoute

    from bentoml._internal import external_typing as ext
    from bentoml._internal.context import ServiceContext
    from bentoml._internal.types import LifecycleHook

R = t.TypeVar("R")


class ContextMiddleware:
    def __init__(self, app: ext.ASGIApp, context: ServiceContext) -> None:
        self.app = app
        self.context = context

    async def __call__(
        self, scope: ext.ASGIScope, receive: ext.ASGIReceive, send: ext.ASGISend
    ) -> None:
        from starlette.requests import Request

        if scope["type"] not in ("http", "websocket"):
            return await self.app(scope, receive, send)

        req = Request(scope, receive, send)
        with self.context.in_request(req):
            await self.app(scope, receive, send)


class ServiceAppFactory(BaseAppFactory):
    @inject
    def __init__(
        self,
        service: Service[t.Any],
        enable_metrics: bool = Provide[
            BentoMLContainer.api_server_config.metrics.enabled
        ],
        traffic: dict[str, t.Any] = Provide[BentoMLContainer.api_server_config.traffic],
    ) -> None:
        from bentoml._internal.runner.container import AutoContainer

        self.service = service
        self.enable_metrics = enable_metrics
        timeout = traffic.get("timeout")
        max_concurrency = traffic.get("max_concurrency")
        super().__init__(timeout=timeout, max_concurrency=max_concurrency)

        self.dispatchers: dict[str, CorkDispatcher[t.Any, t.Any]] = {}
        self._service_instance: t.Any | None = None

        def fallback() -> t.NoReturn:
            raise ServiceUnavailable("process is overloaded")

        for name, method in service.apis.items():
            if not method.batchable:
                continue
            self.dispatchers[name] = CorkDispatcher(
                max_latency_in_ms=method.max_latency_ms,
                max_batch_size=method.max_batch_size,
                fallback=fallback,
                get_batch_size=functools.partial(
                    AutoContainer.get_batch_size, batch_dim=method.batch_dim[0]
                ),
            )

    async def index_page(self, _: Request) -> Response:
        from starlette.responses import FileResponse

        if BentoMLContainer.new_index:
            filename = "main-ui.html"
        else:
            filename = "main-openapi.html"
        return FileResponse(Path(__file__).parent / filename)

    async def openapi_spec_view(self, req: Request) -> Response:
        from starlette.responses import JSONResponse

        try:
            return JSONResponse(self.service.openapi_spec.asdict())
        except Exception:
            log_exception(req, sys.exc_info())
            raise

    async def handle_uncaught_exception(self, req: Request, exc: Exception) -> Response:
        from starlette.responses import JSONResponse

        log_exception(req, sys.exc_info())
        return JSONResponse(
            {"error": "An unexpected error has occurred, please check the server log."},
            status_code=500,
        )

    async def handle_validation_error(self, req: Request, exc: Exception) -> Response:
        from starlette.responses import JSONResponse

        assert isinstance(exc, pydantic.ValidationError)

        data = {
            "error": f"{exc.error_count} validation error for {exc.title}",
            "detail": exc.errors(),
        }
        return JSONResponse(data, status_code=400)

    async def handle_bentoml_exception(self, req: Request, exc: Exception) -> Response:
        from starlette.responses import JSONResponse

        log_exception(req, sys.exc_info())
        assert isinstance(exc, BentoMLException)
        status = exc.error_code.value
        if 400 <= status < 500 and status not in (401, 403):
            return JSONResponse(
                {"error": f"BentoService error handling API request: {exc}"},
                status_code=status,
            )
        else:
            return JSONResponse("", status_code=status)

    def __call__(self, is_main: bool = False) -> Starlette:
        app = super().__call__()

        app.add_exception_handler(
            pydantic.ValidationError, self.handle_validation_error
        )
        app.add_exception_handler(BentoMLException, self.handle_bentoml_exception)
        app.add_exception_handler(Exception, self.handle_uncaught_exception)
        app.add_route("/schema.json", self.schema_view, name="schema")
        if is_main:
            if BentoMLContainer.new_index:
                assets = Path(__file__).parent / "assets"
                app.mount("/assets", StaticFiles(directory=assets), name="assets")
            else:
                from bentoml._internal import server

                assets = Path(server.__file__).parent / "static_content"
                app.mount(
                    "/static_content",
                    StaticFiles(directory=assets),
                    name="static_content",
                )
                app.add_route("/docs.json", self.openapi_spec_view, name="openapi-spec")
            app.add_route("/", self.index_page, name="index")
        for mount_app, path, name in self.service.mount_apps:
            app.mount(app=mount_app, path=path, name=name)
        return app

    @property
    def name(self) -> str:
        return self.service.inner.__name__

    @property
    def middlewares(self) -> list[Middleware]:
        from opentelemetry.instrumentation.asgi import OpenTelemetryMiddleware

        from bentoml._internal.container import BentoMLContainer

        middlewares = super().middlewares + [
            Middleware(ContextMiddleware, context=self.service.context)
        ]

        for middleware_cls, options in self.service.middlewares:
            middlewares.append(Middleware(middleware_cls, **options))

        def client_request_hook(span: Span | None, _scope: dict[str, t.Any]) -> None:
            from bentoml._internal.context import trace_context

            if span is not None:
                trace_context.request_id = span.context.span_id

        middlewares.append(
            Middleware(
                OpenTelemetryMiddleware,
                excluded_urls=BentoMLContainer.tracing_excluded_urls.get(),
                default_span_details=None,
                server_request_hook=None,
                client_request_hook=client_request_hook,
                tracer_provider=BentoMLContainer.tracer_provider.get(),
            )
        )

        if self.enable_metrics:
            from bentoml._internal.server.http.instruments import (
                RunnerTrafficMetricsMiddleware,
            )

            middlewares.append(Middleware(RunnerTrafficMetricsMiddleware))

        access_log_config = BentoMLContainer.api_server_config.logging.access
        if access_log_config.enabled.get():
            from bentoml._internal.server.http.access import AccessLogMiddleware

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

    def create_instance(self) -> None:
        self._service_instance = self.service()

    async def destroy_instance(self) -> None:
        from _bentoml_sdk.service.dependency import cleanup

        await cleanup()
        self._service_instance = None

    async def readyz(self, _: Request) -> Response:
        from starlette.exceptions import HTTPException
        from starlette.responses import PlainTextResponse

        from ..client import RemoteProxy

        if BentoMLContainer.api_server_config.runner_probe.enabled.get():
            dependency_statuses: list[t.Coroutine[None, None, bool]] = []
            for dependency in self.service.dependencies.values():
                real = dependency.get()
                if isinstance(real, RemoteProxy):
                    dependency_statuses.append(real.is_ready())
            runners_ready = all(await asyncio.gather(*dependency_statuses))

            if not runners_ready:
                raise HTTPException(status_code=503, detail="Runners are not ready.")

        return PlainTextResponse("\n", status_code=200)

    @property
    def on_startup(self) -> list[LifecycleHook]:
        return [*super().on_startup, self.create_instance, *self.service.startup_hooks]

    @property
    def on_shutdown(self) -> list[LifecycleHook]:
        return [
            *super().on_shutdown,
            self.destroy_instance,
            *self.service.shutdown_hooks,
        ]

    async def schema_view(self, request: Request) -> Response:
        from starlette.responses import JSONResponse

        schema = self.service.schema()
        return JSONResponse(schema)

    @property
    def routes(self) -> list[BaseRoute]:
        from starlette.routing import Route

        routes = super().routes

        for name, method in self.service.apis.items():
            api_endpoint = functools.partial(self.api_endpoint, name)
            route_path = method.route
            if not route_path.startswith("/"):
                route_path = "/" + route_path
            routes.append(Route(route_path, api_endpoint, methods=["POST"], name=name))
        return routes

    async def batch_infer(
        self, name: str, input_args: tuple[t.Any, ...], input_kwargs: dict[str, t.Any]
    ) -> t.Any:
        method = self.service.apis[name]
        func = getattr(self._service_instance, name)

        async def inner_infer(
            batches: t.Sequence[t.Any], **kwargs: t.Any
        ) -> t.Sequence[t.Any]:
            from starlette.concurrency import run_in_threadpool

            from bentoml._internal.runner.container import AutoContainer
            from bentoml._internal.utils import is_async_callable

            batch, indices = AutoContainer.batches_to_batch(
                batches, method.batch_dim[0]
            )
            if is_async_callable(func):
                result = await func(batch, **kwargs)
            else:
                result = await run_in_threadpool(func, batch, **kwargs)
            return AutoContainer.batch_to_batches(result, indices, method.batch_dim[1])

        arg_names = [k for k in input_kwargs if k not in ("ctx", "context")]
        if input_args:
            if len(input_args) > 1 or len(arg_names) > 0:
                raise TypeError("Batch inference function only accept one argument")
            value = input_args[0]
        else:
            if len(arg_names) != 1:
                raise TypeError("Batch inference function only accept one argument")
            value = input_kwargs.pop(arg_names[0])
        return await self.dispatchers[name](
            functools.partial(inner_infer, **input_kwargs)
        )(value)

    async def api_endpoint(self, name: str, request: Request) -> Response:
        from starlette.concurrency import run_in_threadpool

        from _bentoml_sdk.io_models import ARGS
        from _bentoml_sdk.io_models import KWARGS
        from bentoml._internal.container import BentoMLContainer
        from bentoml._internal.context import trace_context
        from bentoml._internal.utils.http import set_cookies

        from ..serde import ALL_SERDE

        media_type = request.headers.get("Content-Type", "application/json")
        media_type = media_type.split(";")[0].strip()

        method = self.service.apis[name]
        func = getattr(self._service_instance, name)
        ctx = self.service.context
        try:
            serde = ALL_SERDE[media_type]()
            input_data = await method.input_spec.from_http_request(request, serde)
            input_args: tuple[t.Any, ...] = ()
            input_params = {k: getattr(input_data, k) for k in input_data.model_fields}
            if method.ctx_param is not None:
                input_params[method.ctx_param] = ctx
            if ARGS in input_params:
                input_args = tuple(input_params.pop(ARGS))
            if KWARGS in input_params:
                input_params.update(input_params.pop(KWARGS))

            original_func = func
            while hasattr(original_func, "func"):
                original_func = original_func.func

            if method.batchable:
                output = await self.batch_infer(name, input_args, input_params)
            elif inspect.iscoroutinefunction(original_func):
                output = await func(*input_args, **input_params)
            elif inspect.isasyncgenfunction(original_func):
                output = func(*input_args, **input_params)
            else:
                output = await run_in_threadpool(func, *input_args, **input_params)

            response = await method.output_spec.to_http_response(output, serde)
            response.headers.update({"Server": f"BentoML Service/{self.service.name}"})

            if method.ctx_param is not None:
                response.status_code = ctx.response.status_code
                response.headers.update(ctx.response.metadata)
                set_cookies(response, ctx.response.cookies)
            if trace_context.request_id is not None:
                response.headers["X-BentoML-Request-ID"] = str(trace_context.request_id)
            if (
                BentoMLContainer.http.response.trace_id.get()
                and trace_context.trace_id is not None
            ):
                response.headers["X-BentoML-Trace-ID"] = str(trace_context.trace_id)
        finally:
            await request.close()
        return response
