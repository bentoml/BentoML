from __future__ import annotations

import importlib
import inspect
import logging
import os
import sys
import typing as t
from functools import partial

import attr
from pydantic import BaseModel
from simple_di import inject

from ...exceptions import BentoMLException
from ...exceptions import NotFound
from ...grpc.utils import LATEST_PROTOCOL_VERSION
from ...grpc.utils import import_grpc
from ..bento.bento import get_default_svc_readme
from ..context import ServiceContext as Context
from ..features import NEW_IO
from ..io_descriptors import IODescriptor
from ..io_descriptors.base import IOType
from ..models import Model
from ..runner.runner import AbstractRunner
from ..runner.runner import Runner
from ..runner.runner import TritonRunner
from ..tag import Tag
from ..utils import first_not_none
from .inference_api import APIEndpoint
from .inference_api import InferenceAPI

if t.TYPE_CHECKING:
    import grpc

    import bentoml

    from ...grpc.types import AddServicerFn
    from ...grpc.types import ServicerClass
    from ...grpc.v1 import service_pb2_grpc as services
    from .. import external_typing as ext
    from ..bento import Bento
    from ..types import LifecycleHook
    from .openapi.specification import OpenAPISpecification

    ContextFunc = t.Callable[[Context], None | t.Coroutine[t.Any, t.Any, None]]
    HookF = t.TypeVar("HookF", bound=LifecycleHook)
    HookF_ctx = t.TypeVar("HookF_ctx", bound=ContextFunc)
    CB = t.TypeVar("CB", bound=t.Callable)

    class _inference_api_wrapper(t.Generic[IOType]):
        __name__: str

        # fmt: off
        @t.overload
        def __call__(self, func: t.Callable[[IOType], IOType]) -> t.Callable[[IOType], IOType]: ...  # type: ignore (this is considered as stub)
        @t.overload
        def __call__(self, func: t.Callable[[IOType], t.Coroutine[IOType, t.Any, t.Any]]) -> t.Callable[[IOType], t.Coroutine[IOType, t.Any, t.Any]]: ...
        @t.overload
        def __call__(self, func: t.Callable[[IOType, bentoml.Context], IOType]) -> t.Callable[[IOType, bentoml.Context], IOType]: ...
        @t.overload
        def __call__(self, func: t.Callable[[IOType, bentoml.Context], t.Coroutine[IOType, t.Any, t.Any]]) -> t.Callable[[IOType, bentoml.Context], t.Coroutine[IOType, t.Any, t.Any]]: ...
        # fmt: on

else:
    grpc, _ = import_grpc()

logger = logging.getLogger(__name__)


def get_valid_service_name(user_provided_svc_name: str) -> str:
    lower_name = user_provided_svc_name.lower()

    if user_provided_svc_name != lower_name:
        logger.warning(
            "Converting %s to lowercase: %s.", user_provided_svc_name, lower_name
        )

    # Service name must be a valid Tag name; create a dummy tag to use its validation
    Tag(lower_name)
    return lower_name


@attr.define(frozen=False, init=False)
class Service:
    """The service definition is the manifestation of the Service Oriented Architecture
    and the core building block in BentoML where users define the service runtime
    architecture and model serving logic.

    A BentoML service is defined via instantiate this Service class. When creating a
    Service instance, user must provide a Service name and list of runners that are
    required by this Service. The instance can then be used to define InferenceAPIs via
    the `api` decorator.
    """

    name: str
    runners: t.List[Runner | TritonRunner]
    models: t.List[Model]

    # starlette related
    mount_apps: t.List[t.Tuple[ext.ASGIApp, str, str]] = attr.field(
        init=False, factory=list
    )
    middlewares: t.List[
        t.Tuple[t.Type[ext.AsgiMiddleware], t.Dict[str, t.Any]]
    ] = attr.field(init=False, factory=list)

    # gRPC related
    mount_servicers: list[tuple[ServicerClass, AddServicerFn, list[str]]] = attr.field(
        init=False, factory=list
    )
    interceptors: list[partial[grpc.aio.ServerInterceptor]] = attr.field(
        init=False, factory=list
    )
    grpc_handlers: list[grpc.GenericRpcHandler] = attr.field(init=False, factory=list)

    # list of APIs from @svc.api
    apis: t.Dict[str, InferenceAPI[t.Any]] = attr.field(init=False, factory=dict)
    endpoints: t.Dict[str, APIEndpoint] = attr.field(init=False, factory=dict)

    # Tag/Bento are only set when the service was loaded from a bento
    tag: Tag | None = attr.field(init=False, default=None)
    bento: Bento | None = attr.field(init=False, default=None)

    # Working dir and Import path of the service, set when the service was imported
    _working_dir: str | None = attr.field(init=False, default=None)
    _import_str: str | None = attr.field(init=False, default=None)
    _caller_module: str | None = attr.field(init=False, default=None)

    # service context
    context: Context = attr.field(init=False, factory=Context)

    # hooks
    startup_hooks: list[LifecycleHook] = attr.field(init=False, factory=list)
    shutdown_hooks: list[LifecycleHook] = attr.field(init=False, factory=list)
    deployment_hooks: list[LifecycleHook] = attr.field(init=False, factory=list)
    service_str: str = attr.field(init=False, default="BentoML API Service")
    worker_env_map: dict[int, dict[str, t.Any]] | None = attr.field(
        init=False, default=None
    )
    required_workers_num: int | None = attr.field(init=False, default=None)

    def __reduce__(self):
        """
        Make bentoml.Service pickle serializable
        """
        import fs

        from bentoml._internal.bento.bento import Bento
        from bentoml._internal.configuration.containers import BentoMLContainer
        from bentoml._internal.service.loader import load
        from bentoml._internal.service.loader import load_bento_dir

        serialization_strategy = BentoMLContainer.serialization_strategy.get()

        if self.bento:
            if serialization_strategy == "EXPORT_BENTO":
                temp_fs = fs.open_fs("temp:///")
                tmp_path = temp_fs.getsyspath("/")
                bento_path = self.bento.export(tmp_path, output_format="tar")
                content = open(bento_path, "rb").read()

                def load_exported_bento(bento_tag: Tag, content: bytes):
                    tmp_bento_store = BentoMLContainer.tmp_bento_store.get()
                    try:
                        bento = tmp_bento_store.get(bento_tag)
                        return load_bento_dir(bento.path)
                    except NotFound:
                        temp_fs = fs.open_fs("temp:///")
                        temp_fs.writebytes("/import.bento", content)
                        bento = Bento.import_from(
                            temp_fs.getsyspath("/import.bento")
                        ).save(tmp_bento_store)
                        return load_bento_dir(bento.path)

                return (
                    load_exported_bento,
                    (
                        self.bento.tag,
                        content,
                    ),
                )
            elif serialization_strategy == "LOCAL_BENTO":
                return (load, (self.bento.tag,))
            else:
                # serialization_strategy == REMOTE_BENTO
                def get_or_pull(bento_tag: Tag) -> Service:
                    try:
                        return load(bento_tag)
                    except NotFound:
                        pass
                    tmp_bento_store = BentoMLContainer.tmp_bento_store.get()
                    try:
                        bento = tmp_bento_store.get(bento_tag)
                        return load_bento_dir(bento.path)
                    except NotFound:
                        cloud_client = BentoMLContainer.bentocloud_client.get()
                        cloud_client.pull_bento(bento_tag, bento_store=tmp_bento_store)
                        return get_or_pull(bento_tag)

                return (get_or_pull, (self.bento.tag,))
        else:
            from bentoml._internal.service.loader import import_service

            return (import_service, self.get_service_import_origin())

    def __init__(
        self,
        name: str,
        *,
        runners: list[AbstractRunner] | None = None,
        models: list[Model] | None = None,
        caller_depth: int = 1,
    ):
        """Service definition itself. Runners and models can be optionally pass into a ``bentoml.Service``.

        Args:
            name: name of the service
            runners: Optional list of runners to be used with this service.
            models: Optional list of ``bentoml.Model`` to be used with this service.
        """
        name = get_valid_service_name(name)

        # validate runners list contains Runner instances and runner names are unique
        if runners is not None:
            runner_names: t.Set[str] = set()
            for r in runners:
                assert isinstance(
                    r, AbstractRunner
                ), f'Service runners list can only contain bentoml.Runner instances, type "{type(r)}" found.'

                if r.name in runner_names:
                    raise ValueError(
                        f"Found duplicate name `{r.name}` in service runners."
                    )
                runner_names.add(r.name)

        # validate models list contains Model instances
        if models is not None:
            for model in models:
                assert isinstance(
                    model, Model
                ), f'Service models list can only contain bentoml.Model instances, type "{type(model)}" found.'

        self.__attrs_init__(  # type: ignore
            name=name,
            runners=[] if runners is None else runners,
            models=[] if models is None else models,
        )

        # Set import origin info - import_str can not be determined at this stage yet as
        # the variable name is only available in module vars after __init__ is returned
        # get_service_import_origin below will use the _caller_module for retriving the
        # correct import_str for this service
        caller_frame = inspect.currentframe()
        for _ in range(caller_depth):
            assert caller_frame is not None
            caller_frame = caller_frame.f_back
        caller_module = caller_frame.f_globals["__name__"]
        self._caller_module = caller_module
        self._working_dir = os.getcwd()

    def get_service_import_origin(self) -> tuple[str, str]:
        """
        Returns the module name and working directory of the service
        """
        if not self._import_str:
            import_module = self._caller_module
            if import_module == "__main__":
                if hasattr(sys.modules["__main__"], "__file__"):
                    import_module = sys.modules["__main__"].__file__
                else:
                    raise BentoMLException(
                        "Failed to get service import origin, bentoml.Service object defined interactively in console or notebook is not supported"
                    )

            if self._caller_module not in sys.modules:
                raise BentoMLException(
                    "Failed to get service import origin, bentoml.Service object must be defined in a module"
                )

            for name, value in vars(sys.modules[self._caller_module]).items():
                if value is self:
                    self._import_str = f"{import_module}:{name}"
                    break
            else:
                raise BentoMLException(
                    "Failed to get service import origin, bentoml.Service object must be assigned to a variable at module level"
                )

        assert self._working_dir is not None

        return self._import_str, self._working_dir

    def is_service_importable(self) -> bool:
        if self._caller_module == "__main__":
            if not hasattr(sys.modules["__main__"], "__file__"):
                return False

        return True

    if NEW_IO:

        def api(
            self, *, name: str | None, doc: str | None = None, route: str | None = None
        ) -> t.Callable[[CB], CB]:
            def decorator(fn: CB) -> CB:
                self.add_api(fn, name=name, doc=doc, route=route)
                return fn

            return decorator

        def add_api(
            self,
            fn: t.Callable[..., t.Any],
            *,
            name: str | None,
            doc: str | None = None,
            route: str | None = None,
            input_spec: type[BaseModel] | None = None,
            output_spec: type[BaseModel] | None = None,
            stream_output: bool | None = None,
        ) -> None:
            _api = APIEndpoint(
                fn,
                name=name,
                doc=doc,
                route=route,
                input_spec=input_spec,
                output_spec=output_spec,
                stream_output=stream_output,
            )
            if _api.name in self.endpoints:
                raise BentoMLException(
                    f"API {_api.name} is already defined in Service {self.name}"
                )
            self.endpoints[_api.name] = _api

    else:
        # fmt: off
        # case 1: function is not defined, but input and output are
        @t.overload
        def api(self, input: IODescriptor[IOType], output: IODescriptor[IOType]) ->  _inference_api_wrapper[IOType]: ...
        # case 2: the decorator itself with custom routes
        @t.overload
        def api(self, input: IODescriptor[IOType], output: IODescriptor[IOType], *, route: str = ...) ->  _inference_api_wrapper[IOType]: ...
        # fmt: on
        def api(
            self,
            input: IODescriptor[IOType],
            output: IODescriptor[IOType],
            *,
            name: str | None = None,
            doc: str | None = None,
            route: str | None = None,
        ) -> _inference_api_wrapper[IOType]:
            """Decorator for adding InferenceAPI to this service"""

            def decorator(
                fn: _inference_api_wrapper[IOType],
            ) -> _inference_api_wrapper[IOType]:
                _api = InferenceAPI[IOType](
                    name=first_not_none(name, default=fn.__name__),
                    user_defined_callback=fn,
                    input_descriptor=input,
                    output_descriptor=output,
                    doc=doc,
                    route=route,
                )
                if _api.name in self.apis:
                    raise BentoMLException(
                        f"API {_api.name} is already defined in Service {self.name}"
                    )
                self.apis[_api.name] = _api
                return fn

            return t.cast("_inference_api_wrapper[IOType]", decorator)

    def __repr__(self) -> str:
        if self.bento:
            return f'bentoml.Service(tag="{self.tag}", ' f'path="{self.bento.path}")'

        try:
            import_str, working_dir = self.get_service_import_origin()
            return (
                f'bentoml.Service(name="{self.name}", '
                f'import_str="{import_str}", '
                f'working_dir="{working_dir}")'
            )
        except BentoMLException:
            return (
                f'bentoml.Service(name="{self.name}", '
                f'runners=[{",".join([r.name for r in self.runners])}])'
            )

    def __str__(self) -> str:
        return repr(self)

    def __eq__(self, other: t.Any) -> bool:
        if not isinstance(other, Service):
            return NotImplemented
        if self is other:
            return True

        if self.bento and other.bento:
            return self.bento.tag == other.bento.tag

        try:
            return self.get_service_import_origin() == other.get_service_import_origin()
        except BentoMLException:
            return False

    @property
    def doc(self) -> str:
        if self.bento is not None:
            return self.bento.doc

        return get_default_svc_readme(self)

    @property
    def openapi_spec(self) -> OpenAPISpecification:
        from .openapi import generate_spec

        return generate_spec(self)

    def on_startup(self, func: HookF_ctx) -> HookF_ctx:
        self.startup_hooks.append(partial(func, self.context))
        return func

    def on_shutdown(self, func: HookF_ctx) -> HookF_ctx:
        self.shutdown_hooks.append(partial(func, self.context))
        return func

    def on_deployment(self, func: HookF) -> HookF:
        self.deployment_hooks.append(func)
        return func

    def on_asgi_app_startup(self) -> None:
        pass

    def on_asgi_app_shutdown(self) -> None:
        pass

    def on_grpc_server_startup(self) -> None:
        pass

    def on_grpc_server_shutdown(self) -> None:
        pass

    def get_grpc_servicer(
        self, protocol_version: str = LATEST_PROTOCOL_VERSION
    ) -> services.BentoServiceServicer:
        """
        Return a gRPC servicer instance for this service.

        Args:
            protocol_version: The protocol version to use for the gRPC servicer.

        Returns:
            A bento gRPC servicer implementation.
        """
        return importlib.import_module(
            f".grpc.servicer.{protocol_version}",
            package="bentoml._internal.server",
        ).create_bento_servicer(self)

    @property
    def grpc_servicer(self):
        return self.get_grpc_servicer(protocol_version=LATEST_PROTOCOL_VERSION)

    @inject
    def get_asgi_app(self, is_main: bool = False) -> ext.ASGIApp:
        from ..container import BentoMLContainer
        from ..server.uni_app import UnifiedAppFactory

        if is_main:
            enable_metrics = BentoMLContainer.api_server_config.metrics.enabled.get()
            traffic_config = BentoMLContainer.api_server_config.traffic.get().copy()
        else:
            enable_metrics = BentoMLContainer.runners_config.metrics.enabled.get()
            traffic_config = BentoMLContainer.runners_config.traffic.get().copy()

        runner_config = BentoMLContainer.runners_config.get()
        if self.name in runner_config:
            traffic_config.update(runner_config[self.name].get("traffic", {}))

        return UnifiedAppFactory(
            self,
            enable_metrics=enable_metrics,
            timeout=traffic_config["timeout"],
            max_concurrency=traffic_config["max_concurrency"],
        )(is_main)

    @property
    def asgi_app(self) -> "ext.ASGIApp":
        from ..server.http_app import HTTPAppFactory

        return HTTPAppFactory(self)()

    @property
    def schema(self) -> dict[str, t.Any]:
        return {
            "name": self.name,
            "type": "service",
            "routes": [endpoint.asdict() for endpoint in self.endpoints.values()],
        }

    def mount_asgi_app(
        self, app: "ext.ASGIApp", path: str = "/", name: t.Optional[str] = None
    ) -> None:
        self.mount_apps.append((app, path, name))  # type: ignore

    def mount_wsgi_app(
        self, app: ext.WSGIApp, path: str = "/", name: t.Optional[str] = None
    ) -> None:
        # TODO: Migrate to a2wsgi
        from starlette.middleware.wsgi import WSGIMiddleware

        self.mount_apps.append((WSGIMiddleware(app), path, name))  # type: ignore

    def add_asgi_middleware(
        self, middleware_cls: t.Type[ext.AsgiMiddleware], **options: t.Any
    ) -> None:
        self.middlewares.append((middleware_cls, options))

    def mount_grpc_servicer(
        self,
        servicer_cls: ServicerClass,
        add_servicer_fn: AddServicerFn,
        service_names: list[str],
    ) -> None:
        self.mount_servicers.append((servicer_cls, add_servicer_fn, service_names))

    def add_grpc_interceptor(
        self, interceptor_cls: t.Type[grpc.aio.ServerInterceptor], **options: t.Any
    ) -> None:
        from bentoml.exceptions import BadInput

        if not issubclass(interceptor_cls, grpc.aio.ServerInterceptor):
            if isinstance(interceptor_cls, partial):
                if options:
                    logger.debug(
                        "'%s' is a partial class, hence '%s' will be ignored.",
                        interceptor_cls,
                        options,
                    )
                if not issubclass(interceptor_cls.func, grpc.aio.ServerInterceptor):
                    raise BadInput(
                        "'partial' class is not a subclass of 'grpc.aio.ServerInterceptor'."
                    )
                self.interceptors.append(interceptor_cls)
            else:
                raise BadInput(
                    f"{interceptor_cls} is not a subclass of 'grpc.aio.ServerInterceptor'."
                )

        self.interceptors.append(partial(interceptor_cls, **options))

    def add_grpc_handlers(self, handlers: list[grpc.GenericRpcHandler]) -> None:
        self.grpc_handlers.extend(handlers)


def on_load_bento(svc: Service, bento: Bento):
    svc.bento = bento
    svc.tag = bento.info.tag
