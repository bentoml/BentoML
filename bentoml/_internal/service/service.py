import sys
import typing as t
import logging
from typing import TYPE_CHECKING

from ..types import Tag
from ..runner import Runner
from ...exceptions import BentoMLException
from ..bento.bento import _get_default_bento_readme
from .inference_api import InferenceAPI
from ..io_descriptors import IODescriptor
from ..utils.validation import validate_tag_str

if TYPE_CHECKING:
    from starlette.types import ASGIApp
    from starlette.middleware import Middleware
    from starlette.applications import Starlette

    from ..bento import Bento


WSGI_APP = t.Callable[
    [t.Callable[..., t.Any], t.Mapping[str, t.Any]], t.Iterable[bytes]
]

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

    # Name of the service, it is a required parameter for __init__
    name: str
    # Tag/Bento/Version are only applicable if the service was load from a bento
    tag: t.Optional[Tag] = None
    bento: t.Optional["Bento"] = None
    version: t.Optional[str] = None
    # Working dir of the service, set when the service was load from a bento
    _working_dir: t.Optional[str] = None
    # Import path set by .loader.import_service method
    _import_str: t.Optional[str] = None
    # For docs property
    _doc: t.Optional[str] = None

    def __init__(self, name: str, runners: t.Optional[t.List[Runner]] = None):
        lower_name = name.lower()

        if name != lower_name:
            logger.warning(f"converting {name} to lowercase: {lower_name}")

        # Service name must be a valid dns1123 subdomain string
        validate_tag_str(lower_name)
        self.name = lower_name

        if runners is not None:
            assert all(
                isinstance(r, Runner) for r in runners
            ), "Service runners list must only contain runner instances"
            self.runners = {r.name: r for r in runners}
        else:
            self.runners: t.Dict[str, Runner] = {}

        self._mount_apps: t.List[t.Tuple[t.Union["ASGIApp", WSGI_APP], str, str]] = []
        self._middlewares: t.List[t.Tuple[t.Type["Middleware"], t.Any]] = []

    def _on_asgi_app_startup(self) -> None:
        # TODO: initialize Local Runner instances or Runner Clients here
        # TODO(P1): add `@svc.on_startup` decorator for adding user-defined hook
        pass

    def _on_asgi_app_shutdown(self) -> None:
        # TODO(P1): add `@svc.on_shutdown` decorator for adding user-defined hook
        pass

    def __del__(self):
        # working dir was added to sys.path in the .loader.import_service function
        if self._working_dir and sys.path:
            sys.path.remove(self._working_dir)

    def api(
        self,
        input: IODescriptor[t.Any],  # noqa # pylint: disable=redefined-builtin
        output: IODescriptor[t.Any],
        name: t.Optional[str] = None,
        doc: t.Optional[str] = None,
        route: t.Optional[str] = None,
    ) -> t.Callable[[t.Callable[..., t.Any]], t.Callable[..., t.Any]]:
        """Decorator for adding InferenceAPI to this service"""

        D = t.TypeVar("D", bound=t.Callable[..., t.Any])

        def decorator(func: D) -> D:
            from ..io_descriptors import Multipart

            if isinstance(output, Multipart):
                logger.warning(
                    f"Found Multipart as the output of API `{name or func.__name__}`. "
                    "Multipart response is rarely used in the real world,"
                    " less clients/browsers support it. "
                    "Make sure you know what you are doing."
                )
            self._add_inference_api(func, input, output, name, doc, route)
            return func

        return decorator

    def _add_inference_api(
        self,
        func: t.Callable[..., t.Any],
        input: IODescriptor[t.Any],  # noqa # pylint: disable=redefined-builtin
        output: IODescriptor[t.Any],
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
        logger.info("Initializing HTTP server for %s", self)
        from ..server.service_app import ServiceAppFactory

        return ServiceAppFactory(self)()

    def mount_asgi_app(
        self, app: "ASGIApp", path: str = "/", name: t.Optional[str] = None
    ) -> None:
        self._mount_apps.append((app, path, name))  # type: ignore

    def mount_wsgi_app(
        self, app: WSGI_APP, path: str = "/", name: t.Optional[str] = None
    ) -> None:
        from starlette.middleware.wsgi import WSGIMiddleware

        self._mount_apps.append((WSGIMiddleware(app), path, name))  # type: ignore

    def add_asgi_middleware(
        self, middleware_cls: t.Type["Middleware"], **options: t.Any
    ) -> None:
        self._middlewares.append((middleware_cls, options))

    def openapi_doc(self):
        from .openapi import get_service_openapi_doc

        return get_service_openapi_doc(self)

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
                f'runners=[{",".join(self.runners.keys())}])'
            )

    def __repr__(self):
        return self.__str__()

    @property
    def doc(self) -> str:
        if self.bento is not None:
            return self.bento.doc

        return _get_default_bento_readme(self)
