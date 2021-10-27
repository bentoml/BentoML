import sys
import typing as t
from typing import TYPE_CHECKING

from bentoml._internal.io_descriptors import IODescriptor
from bentoml._internal.utils.validation import check_is_dns1123_subdomain
from bentoml.exceptions import BentoMLException

from ..runner import Runner
from .inference_api import InferenceAPI

if TYPE_CHECKING:
    from starlette.applications import Starlette
    from starlette.middleware import Middleware
    from starlette.types import ASGIApp


class Service:
    """The service definition is the manifestation of the Service Oriented Architecture
    and the core building block in BentoML where users define the service runtime
    architecture and model serving logic.

    A BentoML service is defined via instantiate this Service class. When creating a
    Service instance, user must provide a Service name and list of runners that are
    required by this Service. The instance can then be used to define InferenceAPIs via
    the `api` decorator.
    """

    _apis: t.Dict[str, InferenceAPI] = {}
    _runners: t.Dict[str, Runner] = {}

    # Name of the service, it is a required parameter for __init__
    name: str
    # Version of the service, only applicable if the service was load from a bento
    version: t.Optional[str] = None
    # Working dir of the service, set when the service was load from a bento
    _working_dir: t.Optional[str] = None

    def __init__(self, name: str, runners: t.Optional[t.List[Runner]] = None):
        # Service name must be a valid dns1123 subdomain string
        check_is_dns1123_subdomain(name)
        self.name = name

        if runners is not None:
            self._runners = {r.name: r for r in runners}

        self._mount_apps: t.List[t.Tuple["ASGIApp", str, str]] = []
        self._middlewares: t.List["Middleware"] = []

    def on_startup(self) -> None:
        pass

    def on_shutdown(self) -> None:
        # working dir was added to sys.path in the .loader.import_service function
        if self._working_dir:
            sys.path.remove(self._working_dir)

    def api(
        self,
        input_: IODescriptor,
        output: IODescriptor,
        api_name: t.Optional[str] = None,
        api_doc: t.Optional[str] = None,
        route: t.Optional[str] = None,
    ) -> t.Callable[[t.Callable], t.Callable]:
        """Decorator for adding InferenceAPI to this service"""

        def decorator(func: t.Callable) -> t.Callable:
            self._add_inference_api(func, input_, output, api_name, api_doc, route)
            return func

        return decorator

    def _add_inference_api(
        self,
        func: t.Callable,
        input_: IODescriptor,
        output: IODescriptor,
        api_name: t.Optional[str],
        api_doc: t.Optional[str],
        route: t.Optional[str],
    ) -> None:
        api = InferenceAPI(
            name=api_name,
            user_defined_callback=func,
            input_descriptor=input_,
            output_descriptor=output,
            doc=api_doc,
            route=route,
        )

        if api.name in self._apis:
            raise BentoMLException(
                f"API {api_name} is already defined in Service {self.name}"
            )
        self._apis[api.name] = api

    @property
    def asgi_app(self) -> "Starlette":
        from bentoml._internal.server.service_app import ServiceAppFactory

        app_factory = ServiceAppFactory(self)
        return app_factory()

    wsgi_app = asgi_app

    def mount_asgi_app(self, app: "ASGIApp", path: str = "/") -> None:
        self._mount_apps.append((app, path, self.name))

    def add_middleware(
        self, middleware_cls: t.Type["Middleware"], **options: t.Any
    ) -> None:
        self._middlewares.append(middleware_cls(**options))

    def openapi_doc(self):
        from .openapi import get_service_openapi_doc

        return get_service_openapi_doc(self)
