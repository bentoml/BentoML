import logging
import sys
import typing as t
from typing import TYPE_CHECKING

from bentoml._internal.io_descriptors import IODescriptor
from bentoml._internal.utils.validation import validate_tag_name_str
from bentoml.exceptions import BentoMLException

from ..runner import Runner
from .inference_api import InferenceAPI

if TYPE_CHECKING:
    from starlette.applications import Starlette
    from starlette.middleware import Middleware
    from starlette.types import ASGIApp

_WSGI_APP = t.TypeVar("_WSGI_APP")

logger = logging.getLogger(__name__)


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
        lname = name.lower()

        if name != lname:
            logger.warning(f"converting {name} to lowercase: {lname}")

        # Service name must be a valid dns1123 subdomain string
        validate_tag_name_str(lname)
        self.name = lname

        if runners is not None:
            self._runners = {r.name: r for r in runners}

        self._mount_apps: t.List[t.Tuple[t.Union["ASGIApp", _WSGI_APP], str, str]] = []
        self._middlewares: t.List[t.Tuple["Middleware", t.Any]] = []

    def _on_asgi_app_startup(self) -> None:
        # TODO: initialize Local Runner instances or Runner Clients here
        # TODO(P1): add `@svc.on_startup` decorator for adding user-defined hook
        pass

    def _on_asgi_app_shutdown(self) -> None:
        # TODO(P1): add `@svc.on_shutdown` decorator for adding user-defined hook
        pass

    def __del__(self):
        # working dir was added to sys.path in the .loader.import_service function
        if self._working_dir:
            sys.path.remove(self._working_dir)

    def api(
        self,
        input: IODescriptor,
        output: IODescriptor,
        name: t.Optional[str] = None,
        doc: t.Optional[str] = None,
        route: t.Optional[str] = None,
    ) -> t.Callable[[t.Callable], t.Callable]:
        """Decorator for adding InferenceAPI to this service"""

        def decorator(func: t.Callable) -> t.Callable:
            self._add_inference_api(func, input, output, name, doc, route)
            return func

        return decorator

    def _add_inference_api(
        self,
        func: t.Callable,
        input: IODescriptor,
        output: IODescriptor,
        name: t.Optional[str],
        doc: t.Optional[str],
        route: t.Optional[str],
    ) -> None:
        api = InferenceAPI(
            name=name,
            user_defined_callback=func,
            input_descriptor=input,
            output_descriptor=output,
            doc=doc,
            route=route,
        )

        if api.name in self._apis:
            raise BentoMLException(
                f"API {api.name} is already defined in Service {self.name}"
            )
        self._apis[api.name] = api

    @property
    def asgi_app(self) -> "Starlette":
        from bentoml._internal.server.service_app import ServiceAppFactory

        return ServiceAppFactory(self)()

    def mount_asgi_app(self, app: "ASGIApp", path: str = "/", name: str = None) -> None:
        self._mount_apps.append((app, path, name))

    def mount_wsgi_app(self, app: _WSGI_APP, path: str = "/", name: str = None) -> None:
        from starlette.middleware.wsgi import WSGIMiddleware

        self._mount_apps.append((WSGIMiddleware(app), path, name))

    def add_agsi_middleware(
        self, middleware_cls: t.Type["Middleware"], **options: t.Any
    ) -> None:
        self._middlewares.append((middleware_cls, options))

    def openapi_doc(self):
        from .openapi import get_service_openapi_doc

        return get_service_openapi_doc(self)
