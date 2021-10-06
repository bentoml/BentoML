from typing import Any, Dict, List, Optional

from bentoml._internal.io_descriptors import IODescriptor
from bentoml._internal.utils.validation import check_is_dns1123_subdomain
from bentoml.exceptions import BentoMLException

from ..runner import Runner
from .inference_api import InferenceAPI


class Service:
    """
    Default Service/Bento description goes here
    """

    _apis: Dict[str, InferenceAPI] = {}
    _runners: Dict[str, Runner] = {}

    # Name of the service, it is a required parameter for __init__
    name: str
    # Version of the service, only applicable if the service was load from a bento
    version: Optional[str] = None

    def __init__(self, name: str, runners: Optional[List[Runner]] = None):
        # Service name must be a valid dns1123 subdomain string
        check_is_dns1123_subdomain(name)
        self.name = name

        if runners is not None:
            self._runners = {r.name: r for r in runners}

    def api(
        self,
        input: IODescriptor,
        output: IODescriptor,
        api_name: Optional[str] = None,
        api_doc: Optional[str] = None,
        route: Optional[str] = None,
    ):
        """Decorate a user defined function to make it an InferenceAPI of this service"""

        def decorator(func):
            self._add_inference_api(func, input, output, api_name, api_doc, route)

        return decorator

    def _add_inference_api(
        self,
        func: callable,
        input: IODescriptor,
        output: IODescriptor,
        api_name: Optional[str],
        api_doc: Optional[str],
        route: Optional[str],
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
