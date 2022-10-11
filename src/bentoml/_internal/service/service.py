from __future__ import annotations

import typing as t
import logging
from typing import TYPE_CHECKING
from functools import partial

import attr

from bentoml.exceptions import BentoMLException

from ..tag import Tag
from ..models import Model
from ..runner import Runner
from ..bento.bento import get_default_svc_readme
from .inference_api import InferenceAPI
from ..io_descriptors import IODescriptor

if TYPE_CHECKING:
    import grpc

    from bentoml.grpc.types import AddServicerFn
    from bentoml.grpc.types import ServicerClass

    from .. import external_typing as ext
    from ..bento import Bento
    from ..server.grpc.servicer import Servicer
    from .openapi.specification import OpenAPISpecification
else:
    from bentoml.grpc.utils import import_grpc

    grpc, _ = import_grpc()

logger = logging.getLogger(__name__)


def add_inference_api(
    svc: Service,
    func: t.Callable[..., t.Any],
    input: IODescriptor[t.Any],  # pylint: disable=redefined-builtin
    output: IODescriptor[t.Any],
    name: t.Optional[str],
    doc: t.Optional[str],
    route: t.Optional[str],
) -> None:
    api = InferenceAPI(
        name=name if name is not None else func.__name__,
        user_defined_callback=func,
        input_descriptor=input,
        output_descriptor=output,
        doc=doc,
        route=route,
    )

    if api.name in svc.apis:
        raise BentoMLException(
            f"API {api.name} is already defined in Service {svc.name}"
        )

    svc.apis[api.name] = api


def get_valid_service_name(user_provided_svc_name: str) -> str:
    lower_name = user_provided_svc_name.lower()

    if user_provided_svc_name != lower_name:
        logger.warning(
            f"converting {user_provided_svc_name} to lowercase: {lower_name}"
        )

    # Service name must be a valid Tag name; create a dummy tag to use its validation
    Tag(lower_name)
    return lower_name


@attr.define(frozen=True)
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
    runners: t.List[Runner]
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
    apis: t.Dict[str, InferenceAPI] = attr.field(init=False, factory=dict)

    # Tag/Bento are only set when the service was loaded from a bento
    tag: Tag | None = attr.field(init=False, default=None)
    bento: Bento | None = attr.field(init=False, default=None)

    # Working dir and Import path of the service, set when the service was imported
    _working_dir: str | None = attr.field(init=False, default=None)
    _import_str: str | None = attr.field(init=False, default=None)

    def __init__(
        self,
        name: str,
        *,
        runners: t.List[Runner] | None = None,
        models: t.List[Model] | None = None,
    ):
        """

        Args:
            name:
            runners:
            models:
        """
        name = get_valid_service_name(name)

        # validate runners list contains Runner instances and runner names are unique
        if runners is not None:
            runner_names: t.Set[str] = set()
            for r in runners:
                assert isinstance(
                    r, Runner
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

    def api(
        self,
        input: IODescriptor[t.Any],  # pylint: disable=redefined-builtin
        output: IODescriptor[t.Any],
        name: t.Optional[str] = None,
        doc: t.Optional[str] = None,
        route: t.Optional[str] = None,
    ) -> t.Callable[[t.Callable[..., t.Any]], t.Callable[..., t.Any]]:
        """Decorator for adding InferenceAPI to this service"""

        D = t.TypeVar("D", bound=t.Callable[..., t.Any])

        def decorator(func: D) -> D:
            add_inference_api(self, func, input, output, name, doc, route)
            return func

        return decorator

    def __str__(self):
        if self.bento:
            return f'bentoml.Service(tag="{self.tag}", ' f'path="{self.bento.path}")'
        elif self._import_str and self._working_dir:
            return (
                f'bentoml.Service(name="{self.name}", '
                f'import_str="{self._import_str}", '
                f'working_dir="{self._working_dir}")'
            )
        else:
            return (
                f'bentoml.Service(name="{self.name}", '
                f'runners=[{",".join([r.name for r in self.runners])}])'
            )

    def __repr__(self):
        return self.__str__()

    @property
    def doc(self) -> str:
        if self.bento is not None:
            return self.bento.doc

        return get_default_svc_readme(self)

    @property
    def openapi_spec(self) -> OpenAPISpecification:
        from .openapi import generate_spec

        return generate_spec(self)

    def on_asgi_app_startup(self) -> None:
        pass

    def on_asgi_app_shutdown(self) -> None:
        pass

    def on_grpc_server_startup(self) -> None:
        pass

    def on_grpc_server_shutdown(self) -> None:
        pass

    @property
    def grpc_servicer(self) -> Servicer:
        from ..server.grpc_app import GRPCAppFactory

        return GRPCAppFactory(self)()

    @property
    def asgi_app(self) -> "ext.ASGIApp":
        from ..server.http_app import HTTPAppFactory

        return HTTPAppFactory(self)()

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
    object.__setattr__(svc, "bento", bento)
    object.__setattr__(svc, "tag", bento.info.tag)


def on_import_svc(svc: Service, working_dir: str, import_str: str):
    object.__setattr__(svc, "_working_dir", working_dir)
    object.__setattr__(svc, "_import_str", import_str)
