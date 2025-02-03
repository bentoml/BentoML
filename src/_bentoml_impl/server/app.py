from __future__ import annotations

import asyncio
import functools
import inspect
import logging
import math
import os
import typing as t
from pathlib import Path

import anyio
import anyio.to_thread
from simple_di import Provide
from simple_di import inject
from starlette.middleware import Middleware
from starlette.responses import JSONResponse
from starlette.responses import Response
from starlette.staticfiles import StaticFiles

from _bentoml_sdk import Service
from _bentoml_sdk.service import set_current_service
from bentoml._internal.container import BentoMLContainer
from bentoml._internal.marshal.dispatcher import CorkDispatcher
from bentoml._internal.resource import system_resources
from bentoml._internal.server.base_app import BaseAppFactory
from bentoml._internal.server.http_app import log_exception
from bentoml._internal.utils.metrics import exponential_buckets
from bentoml.exceptions import BentoMLException
from bentoml.exceptions import ServiceUnavailable

from ..tasks import ResultStatus
from ..tasks import Sqlite3Store
from .mount import PassiveMount

if t.TYPE_CHECKING:
    from opentelemetry.sdk.trace import Span
    from prometheus_client import Histogram
    from starlette.applications import Starlette
    from starlette.requests import Request
    from starlette.routing import BaseRoute

    from bentoml._internal import external_typing as ext
    from bentoml._internal.context import ServiceContext
    from bentoml._internal.types import LifecycleHook

R = t.TypeVar("R")
logger = logging.getLogger("bentoml.server")
RESULT_STORE_ENV = "BENTOML_RESULT_STORE"


class ContextMiddleware:
    def __init__(self, app: ext.ASGIApp, context: ServiceContext) -> None:
        self.app = app
        self.context = context

    async def __call__(
        self, scope: ext.ASGIScope, receive: ext.ASGIReceive, send: ext.ASGISend
    ) -> None:
        from starlette.requests import Request

        if scope["type"] not in ("http",):
            return await self.app(scope, receive, send)

        req = Request(scope, receive, send)
        with self.context.in_request(req):
            await self.app(scope, receive, send)


class ServiceAppFactory(BaseAppFactory):
    @inject
    def __init__(
        self,
        service: Service[t.Any],
        is_main: bool = False,
        enable_metrics: bool = Provide[
            BentoMLContainer.api_server_config.metrics.enabled
        ],
        services: dict[str, t.Any] = Provide[BentoMLContainer.config.services],
        # traffic: dict[str, t.Any] = Provide[BentoMLContainer.api_server_config.traffic],
        enable_access_control: bool = Provide[BentoMLContainer.http.cors.enabled],
        access_control_options: dict[str, list[str] | str | int] = Provide[
            BentoMLContainer.access_control_options
        ],
    ) -> None:
        from bentoml._internal.runner.container import AutoContainer

        self.service = service
        self.enable_metrics = enable_metrics
        self.is_main = is_main
        config = services[service.name]
        traffic = config.get("traffic")
        workers = config.get("workers")
        timeout = traffic.get("timeout")
        max_concurrency = traffic.get("max_concurrency")
        self.enable_access_control = enable_access_control
        self.access_control_options = access_control_options
        # max_concurrency per worker is the max_concurrency per service divided by the number of workers
        num_workers = 1
        if workers:
            if (workers := config["workers"]) == "cpu_count":
                srs = system_resources()
                num_workers = int(srs["cpu"])
            else:  # workers is a number
                num_workers = workers
        super().__init__(
            timeout=timeout,
            max_concurrency=max_concurrency
            if not max_concurrency
            else math.ceil(max_concurrency / num_workers),
        )

        self.dispatchers: dict[str, CorkDispatcher[t.Any, t.Any]] = {}
        self._service_instance: t.Any | None = None
        self._limiter: anyio.CapacityLimiter | None = None

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
                batch_dim=method.batch_dim,
            )

    @functools.cached_property
    def adaptive_batch_size_hist(self) -> Histogram:
        metrics_client = BentoMLContainer.metrics_client.get()
        max_max_batch_size = max(
            (
                method.max_batch_size
                for method in self.service.apis.values()
                if method.batchable
            ),
            default=100,
        )

        return metrics_client.Histogram(
            namespace="bentoml_service",
            name="adaptive_batch_size",
            documentation="Service adaptive batch size",
            labelnames=[
                "runner_name",
                "worker_index",
                "method_name",
                "service_version",
                "service_name",
            ],
            buckets=exponential_buckets(1, 2, max_max_batch_size),
        )

    async def index_page(self, _: Request) -> Response:
        from starlette.responses import FileResponse

        if BentoMLContainer.new_index:
            filename = "main-ui.html"
        else:
            filename = "main-openapi.html"
        return FileResponse(Path(__file__).parent / filename)

    async def openapi_spec_view(self, req: Request) -> Response:
        try:
            return JSONResponse(self.service.openapi_spec.asdict())
        except Exception:
            log_exception(req)
            return JSONResponse(
                {"error": "Failed to generate OpenAPI spec"}, status_code=500
            )

    def __call__(self) -> Starlette:
        app = super().__call__()
        app.add_route("/schema.json", self.schema_view, name="schema")

        for mount_app, path, name in self.service.mount_apps:
            app.router.routes.append(PassiveMount(path, mount_app, name=name))

        if self.is_main:
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

        return app

    @property
    def name(self) -> str:
        return self.service.name

    @property
    def middlewares(self) -> list[Middleware]:
        from bentoml._internal.container import BentoMLContainer

        middlewares: list[Middleware] = []
        # TrafficMetrics middleware should be the first middleware
        if self.enable_metrics:
            from bentoml._internal.server.http.instruments import (
                RunnerTrafficMetricsMiddleware,
            )

            middlewares.append(
                Middleware(RunnerTrafficMetricsMiddleware, namespace="bentoml_service")
            )

        # OpenTelemetry middleware
        def server_request_hook(span: Span, _scope: dict[str, t.Any]) -> None:
            from bentoml._internal.context import trace_context

            if span.context is not None:
                trace_context.request_id = span.context.span_id

        from opentelemetry.instrumentation.asgi import OpenTelemetryMiddleware

        middlewares.append(
            Middleware(
                OpenTelemetryMiddleware,
                excluded_urls=BentoMLContainer.tracing_excluded_urls.get(),
                default_span_details=None,
                server_request_hook=server_request_hook,
                client_request_hook=None,
                tracer_provider=BentoMLContainer.tracer_provider.get(),
            )
        )
        # AccessLog middleware
        access_log_config = BentoMLContainer.api_server_config.logging.access
        if access_log_config.enabled.get():
            from bentoml._internal.server.http.access import AccessLogMiddleware

            middlewares.append(
                Middleware(
                    AccessLogMiddleware,
                    has_request_content_length=access_log_config.request_content_length.get(),
                    has_request_content_type=access_log_config.request_content_type.get(),
                    has_response_content_length=access_log_config.response_content_length.get(),
                    has_response_content_type=access_log_config.response_content_type.get(),
                    skip_paths=access_log_config.skip_paths.get(),
                )
            )
        # TimeoutMiddleware and MaxConcurrencyMiddleware
        middlewares.extend(super().middlewares)
        for middleware_cls, options in self.service.middlewares:
            middlewares.append(Middleware(middleware_cls, **options))
        # CORS middleware
        if self.enable_access_control:
            assert self.access_control_options.get("allow_origins") is not None, (
                "To enable cors, access_control_allow_origin must be set"
            )

            from starlette.middleware.cors import CORSMiddleware

            middlewares.append(
                Middleware(CORSMiddleware, **self.access_control_options)
            )

        # ContextMiddleware
        middlewares.append(Middleware(ContextMiddleware, context=self.service.context))
        return middlewares

    async def create_instance(self, app: Starlette) -> None:
        from ..client import RemoteProxy

        self._service_instance = self.service()
        self.service.gradio_app_startup_hook(max_concurrency=self.max_concurrency)
        logger.info("Service %s initialized", self.service.name)

        # Call on_startup hook with optional ctx or context parameter
        for name, member in vars(self.service.inner).items():
            if callable(member) and getattr(member, "__bentoml_startup_hook__", False):
                logger.info("Running startup hook: %s", name)
                result = getattr(
                    self._service_instance, name
                )()  # call the bound method
                if inspect.isawaitable(result):
                    await result
                    logger.info("Completed async startup hook: %s", name)
                else:
                    logger.info("Completed startup hook: %s", name)
        if deployment_url := os.getenv("BENTOCLOUD_DEPLOYMENT_URL"):
            proxy = RemoteProxy(
                deployment_url, service=self.service, media_type="application/json"
            )
        else:
            proxy = RemoteProxy("http://localhost:3000", service=self.service, app=app)
        self._service_instance.__self_proxy__ = proxy  # type: ignore[attr-defined]
        self._service_instance.to_async = proxy.to_async  # type: ignore[attr-defined]
        self._service_instance.to_sync = proxy.to_sync  # type: ignore[attr-defined]
        set_current_service(self._service_instance)
        store_path = BentoMLContainer.result_store_file.get()
        self._result_store = Sqlite3Store(store_path)
        await self._result_store.__aenter__()

    @inject
    def _add_response_headers(
        self,
        resp: Response,
        logging_format: dict[str, str] = Provide[BentoMLContainer.logging_formatting],
    ) -> None:
        from bentoml._internal.context import trace_context

        resp.headers.update({"Server": f"BentoML Service/{self.service.name}"})
        if trace_context.request_id is not None:
            resp.headers["X-BentoML-Request-ID"] = format(
                trace_context.request_id, logging_format["span_id"]
            )
        if (
            BentoMLContainer.http.response.trace_id.get()
            and trace_context.trace_id is not None
        ):
            resp.headers["X-BentoML-Trace-ID"] = format(
                trace_context.trace_id, logging_format["trace_id"]
            )

    async def destroy_instance(self, _: Starlette) -> None:
        from _bentoml_sdk.service.dependency import cleanup

        from ..client import RemoteProxy

        # Call on_shutdown hook with optional ctx or context parameter
        for name, member in vars(self.service.inner).items():
            if callable(member) and getattr(member, "__bentoml_shutdown_hook__", False):
                logger.info("Running cleanup hook: %s", name)
                result = getattr(
                    self._service_instance, name
                )()  # call the bound method
                if inspect.isawaitable(result):
                    await result
                    logger.info("Completed async cleanup hook: %s", name)
                else:
                    logger.info("Completed cleanup hook: %s", name)

        await cleanup()
        own_proxy = getattr(self._service_instance, "__self_proxy__", None)
        if isinstance(own_proxy, RemoteProxy):
            await own_proxy.close()
        logger.info("Service instance cleanup finalized")
        self._service_instance = None
        set_current_service(None)
        await self._result_store.__aexit__(None, None, None)

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
        return [*super().on_startup, self.create_instance]

    @property
    def on_shutdown(self) -> list[LifecycleHook]:
        return [*super().on_shutdown, self.destroy_instance]

    async def schema_view(self, request: Request) -> Response:
        schema = self.service.schema()
        return JSONResponse(schema)

    async def get_status(self, request: Request) -> Response:
        task_id = request.query_params.get("task_id")
        if task_id is None:
            resp = JSONResponse({"error": "task_id is required"}, status_code=400)
            self._add_response_headers(resp)
            return resp
        try:
            status = await self._result_store.get_status(task_id)
        except KeyError:
            resp = JSONResponse({"error": "task_id not found"}, status_code=404)
        except Exception as e:
            log_exception(request)
            resp = JSONResponse({"error": str(e)}, status_code=500)
        else:
            resp = JSONResponse(status.to_json())
        self._add_response_headers(resp)
        return resp

    async def get_result(self, request: Request) -> Response:
        task_id = request.query_params.get("task_id")
        if task_id is None:
            resp = JSONResponse({"error": "task_id is required"}, status_code=400)
            self._add_response_headers(resp)
            return resp
        try:
            row = await self._result_store.get(task_id)
        except KeyError:
            resp = JSONResponse({"error": "task_id not found"}, status_code=404)
        except RuntimeError:
            resp = JSONResponse(
                {"error": f"task {task_id} is not completed yet"}, status_code=400
            )
        except Exception as e:
            log_exception(request)
            resp = JSONResponse({"error": str(e)}, status_code=500)
        else:
            resp = row.result
        self._add_response_headers(resp)
        return resp

    async def retry_task(self, request: Request) -> Response:
        task_id = request.query_params.get("task_id")
        if task_id is None:
            resp = JSONResponse({"error": "task_id is required"}, status_code=400)
            self._add_response_headers(resp)
            return resp
        try:
            row = await self._result_store.get(task_id)
        except KeyError:
            resp = JSONResponse({"error": "task_id not found"}, status_code=404)
        except RuntimeError:
            resp = JSONResponse(
                {"error": f"task {task_id} is not completed yet"}, status_code=400
            )
        else:
            resp = await self.submit_task(row.name, row.input)
        self._add_response_headers(resp)
        return resp

    async def cancel_task(self, request: Request) -> Response:
        task_id = request.query_params.get("task_id")
        if task_id is None:
            resp = JSONResponse({"error": "task_id is required"}, status_code=400)
            self._add_response_headers(resp)
            return resp
        await self._result_store.set_status(task_id, ResultStatus.CANCELLED)
        resp = JSONResponse(
            {"error": "task cancellation is not supported in local development server"},
            status_code=400,
        )
        self._add_response_headers(resp)
        return resp

    async def _run_task(self, task_id: str, name: str, request: Request) -> None:
        try:
            self.service.context.request.state.task_id = task_id
            resp = await self.api_endpoint_wrapper(name, request)
            await self._result_store.set_result(
                task_id,
                resp,
                ResultStatus.SUCCESS
                if resp.status_code < 400
                else ResultStatus.FAILURE,
            )
        except Exception:
            logger.exception("Task(%s) %s failed", name, task_id)
        else:
            logger.info("Task(%s) %s is completed", name, task_id)

    async def submit_task(self, name: str, request: Request) -> Response:
        from starlette.background import BackgroundTask

        try:
            task_id = await self._result_store.new_entry(name, request)
        except Exception as e:
            log_exception(request)
            resp = JSONResponse({"error": str(e)}, status_code=500)
            self._add_response_headers(resp)
            return resp
        else:
            logger.info("Task(%s) %s is submitted", name, task_id)
            resp = JSONResponse(
                {"task_id": task_id, "status": ResultStatus.IN_PROGRESS.value}
            )
            resp.background = BackgroundTask(self._run_task, task_id, name, request)
            self._add_response_headers(resp)
            return resp

    @property
    def routes(self) -> list[BaseRoute]:
        from starlette.routing import Route

        routes = super().routes

        for name, method in self.service.apis.items():
            api_endpoint = functools.partial(self.api_endpoint_wrapper, name)
            route_path = method.route
            if not route_path.startswith("/"):
                route_path = "/" + route_path
            routes.append(Route(route_path, api_endpoint, methods=["POST"], name=name))
            if method.is_task:
                routes.append(
                    Route(
                        f"{route_path}/submit",
                        functools.partial(self.submit_task, name),
                        methods=["POST"],
                        name=f"{name}_submit",
                    )
                )
                routes.append(
                    Route(
                        f"{route_path}/status",
                        self.get_status,
                        methods=["GET"],
                        name=f"{name}_status",
                    )
                )
                routes.append(
                    Route(f"{route_path}/get", self.get_result, methods=["GET"])
                )
                routes.append(
                    Route(f"{route_path}/retry", self.retry_task, methods=["POST"])
                )
                routes.append(
                    Route(f"{route_path}/cancel", self.cancel_task, methods=["PUT"])
                )
        return routes

    async def _to_thread(
        self,
        func: t.Callable[..., R],
        *args: t.Any,
        **kwargs: t.Any,
    ) -> R:
        if self._limiter is None:
            threads = self.service.config.get("threads", 1)
            self._limiter = anyio.CapacityLimiter(threads)
        func = functools.partial(func, *args, **kwargs)
        output = await anyio.to_thread.run_sync(func, limiter=self._limiter)
        return output

    async def batch_infer(
        self, name: str, input_args: tuple[t.Any, ...], input_kwargs: dict[str, t.Any]
    ) -> t.Any:
        method = self.service.apis[name]
        func = getattr(self._service_instance, name).local

        async def inner_infer(
            batches: t.Sequence[t.Any], **kwargs: t.Any
        ) -> t.Sequence[t.Any]:
            from bentoml._internal.context import server_context
            from bentoml._internal.runner.container import AutoContainer
            from bentoml._internal.utils import is_async_callable

            if self.enable_metrics:
                self.adaptive_batch_size_hist.labels(  # type: ignore
                    runner_name=self.service.name,
                    worker_index=server_context.worker_index,
                    method_name=name,
                    service_version=server_context.bento_version,
                    service_name=server_context.bento_name,
                ).observe(len(batches))

            if len(batches) == 0:
                return []

            batch, indices = AutoContainer.batches_to_batch(
                batches, method.batch_dim[0]
            )
            if is_async_callable(func):
                result = await func(batch, **kwargs)
            else:
                result = await self._to_thread(func, batch, **kwargs)
            return AutoContainer.batch_to_batches(result, indices, method.batch_dim[1])

        arg_names = [k for k in input_kwargs if k != method.ctx_param]
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

    async def api_endpoint_wrapper(self, name: str, request: Request) -> Response:
        from pydantic import ValidationError

        try:
            resp = await self.api_endpoint(name, request)
        except ValidationError as exc:
            log_exception(request)
            data = {
                "error": f"{exc.error_count()} validation error for {exc.title}",
                "detail": exc.errors(include_context=False, include_input=False),
            }
            resp = JSONResponse(data, status_code=400)
        except BentoMLException as exc:
            log_exception(request)
            status = exc.error_code.value
            if status in (401, 403):
                detail = {
                    "error": "Authorization error",
                }
            elif status >= 500:
                detail = {
                    "error": "An unexpected error has occurred, please check the server log."
                }
            else:
                detail = ({"error": str(exc)},)
            resp = JSONResponse(detail, status_code=status)
        except Exception:
            log_exception(request)
            resp = JSONResponse(
                {
                    "error": "An unexpected error has occurred, please check the server log."
                },
                status_code=500,
            )
        self._add_response_headers(resp)
        ctx = self.service.context
        if resp.background is not None:
            ctx.response.background.tasks.append(resp.background)
        # clean the request resources after the response is consumed.
        ctx.response.background.add_task(request.close)
        resp.background = ctx.response.background
        return resp

    async def api_endpoint(self, name: str, request: Request) -> Response:
        from _bentoml_sdk.io_models import ARGS
        from _bentoml_sdk.io_models import KWARGS
        from bentoml._internal.utils import get_original_func
        from bentoml._internal.utils.http import set_cookies

        from ..serde import ALL_SERDE

        media_type = request.headers.get("Content-Type", "application/json")
        media_type = media_type.split(";")[0].strip()

        method = self.service.apis[name]
        func = getattr(self._service_instance, name).local
        ctx = self.service.context
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

        original_func = get_original_func(func)

        if method.batchable:
            output = await self.batch_infer(name, input_args, input_params)
        elif inspect.iscoroutinefunction(original_func):
            output = await func(*input_args, **input_params)
        elif inspect.isasyncgenfunction(original_func):
            output = func(*input_args, **input_params)
        elif inspect.isgeneratorfunction(original_func):

            async def inner() -> t.AsyncGenerator[t.Any, None]:
                gen = func(*input_args, **input_params)
                while True:
                    try:
                        yield await self._to_thread(next, gen)
                    except StopIteration:
                        break
                    except RuntimeError as e:
                        if "StopIteration" in str(e):
                            break
                        raise

            output = inner()
        else:
            output = await self._to_thread(func, *input_args, **input_params)

        if isinstance(output, Response):
            response = output
        else:
            response = await method.output_spec.to_http_response(output, serde)

        if method.ctx_param is not None:
            response.status_code = ctx.response.status_code
            response.headers.update(ctx.response.metadata)
            set_cookies(response, ctx.response.cookies)
        return response
