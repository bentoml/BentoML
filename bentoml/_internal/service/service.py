import sys
import typing as t
import logging
from typing import TYPE_CHECKING

from bentoml import Runner
from bentoml import SimpleRunner
from bentoml.io import IODescriptor
from bentoml.exceptions import BentoMLException

from ..tag import Tag
from ..bento.bento import get_default_bento_readme
from .inference_api import InferenceAPI

if TYPE_CHECKING:
    from .. import external_typing as ext
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

    apis: t.Dict[str, InferenceAPI] = {}

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

    def __init__(
        self,
        name: str,
        runners: "t.Optional[t.List[t.Union[Runner, SimpleRunner]]]" = None,
    ):
        lower_name = name.lower()

        if name != lower_name:
            logger.warning(f"converting {name} to lowercase: {lower_name}")

        # Service name must be a valid Tag name; create a dummy tag to use its validation
        Tag(lower_name)
        self.name = lower_name

        if runners is not None:
            self.runners = {}
            for r in runners:
                if r.name in self.runners:
                    raise ValueError(
                        f"Found duplicate name `{r.name}` in service runners."
                    )
                assert isinstance(
                    r, (Runner, SimpleRunner)
                ), "Service runners list must only contain runner instances"
                self.runners[r.name] = r
        else:
            self.runners: t.Dict[str, "t.Union[Runner, SimpleRunner]"] = {}

        self.mount_apps: t.List[t.Tuple["ext.ASGIApp", str, str]] = []
        self.middlewares: t.List[
            t.Tuple[t.Type["ext.AsgiMiddleware"], t.Dict[str, t.Any]]
        ] = []

    def on_asgi_app_startup(self) -> None:
        # TODO: initialize Local Runner instances or Runner Clients here
        # TODO(P1): add `@svc.on_startup` decorator for adding user-defined hook
        pass

    def on_asgi_app_shutdown(self) -> None:
        # TODO(P1): add `@svc.on_shutdown` decorator for adding user-defined hook
        pass

    def __del__(self):
        # working dir was added to sys.path in the .loader.import_service function
        if self._working_dir and sys.path:
            sys.path.remove(self._working_dir)

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
        input: IODescriptor[t.Any],  # pylint: disable=redefined-builtin
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

        if api.name in self.apis:
            raise BentoMLException(
                f"API {api.name} is already defined in Service {self.name}"
            )
        self.apis[api.name] = api

    @property
    def asgi_app(self) -> "ext.ASGIApp":
        from ..server.service_app import ServiceAppFactory

        return ServiceAppFactory(self)()

    def mount_asgi_app(
        self, app: "ext.ASGIApp", path: str = "/", name: t.Optional[str] = None
    ) -> None:
        self.mount_apps.append((app, path, name))  # type: ignore

    def mount_wsgi_app(
        self, app: WSGI_APP, path: str = "/", name: t.Optional[str] = None
    ) -> None:
        from starlette.middleware.wsgi import WSGIMiddleware

        self.mount_apps.append((WSGIMiddleware(app), path, name))  # type: ignore

    def add_asgi_middleware(
        self, middleware_cls: t.Type["ext.AsgiMiddleware"], **options: t.Any
    ) -> None:
        self.middlewares.append((middleware_cls, options))

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

        return get_default_bento_readme(self)
