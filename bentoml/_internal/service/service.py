import sys
import typing as t

from bentoml._internal.io_descriptors import IODescriptor
from bentoml._internal.utils.validation import check_is_dns1123_subdomain
from bentoml.exceptions import BentoMLException

from ..runner import Runner
from .inference_api import InferenceAPI


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

    def __del__(self):
        # working dir was added to sys.path in the .loader.import_service function
        if self._working_dir:
            sys.path.remove(self._working_dir)

    def api(
        self,
        input: IODescriptor,
        output: IODescriptor,
        api_name: t.Optional[str] = None,
        api_doc: t.Optional[str] = None,
        route: t.Optional[str] = None,
    ):
        """Decorator for adding InferenceAPI to this service"""

        def decorator(func):
            self._add_inference_api(func, input, output, api_name, api_doc, route)

        return decorator

    def _add_inference_api(
        self,
        func: callable,
        input: IODescriptor,
        output: IODescriptor,
        api_name: t.Optional[str],
        api_doc: t.Optional[str],
        route: t.Optional[str],
    ):
        api = InferenceAPI(
            name=api_name,
            user_defined_callback=func,
            input_descriptor=input,
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
    def _asgi_app(self):
        return self._app

    @property
    def _wsgi_app(self):
        return self._app

    def mount_asgi_app(self, app, path=None):
        self._app.mount(app, path=path)

    def add_middleware(self, middleware, *args, **kwargs):
        self._app

    def openapi_doc(self):
        from .openapi import get_service_openapi_doc

        return get_service_openapi_doc(self)
