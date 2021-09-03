from typing import Dict, List, Optional

from ...exceptions import BentoMLException, InvalidArgument
from ..io_descriptors import IODescriptor
from ..utils.validation import check_is_dns1123_subdomain
from .inference_api import InferenceAPI
from .runner import _Runner


class Service:
    apis: Dict[str, InferenceAPI] = []
    runners: List[_Runner] = []

    # version is only set when load from a bento
    version: Optional[str] = None

    def __init__(self, name: str, runners: Optional[List[_Runner]] = None):
        # Service name must be a valid dns1123 subdomain string
        check_is_dns1123_subdomain(name)
        self.name = name

        if runners is not None:
            self.runners = runners

    def api(
        self,
        input: IODescriptor,
        output: IODescriptor,
        api_name: Optional[str] = None,
        api_doc: Optional[str] = None,
        route: Optional[str] = None,
    ):
        """
        Decorate a user defined function to make it an InferenceAPI of this service
        """
        if not api_name.isidentifier():
            raise InvalidArgument("")

        def decorator(func):
            _api_name = func.__name__ if api_name is None else api_name
            _route = _api_name if route is None else route

            if _api_name in self.apis:
                raise BentoMLException(
                    f"API {api_name} is already defined in Service {self.name}"
                )

            api = InferenceAPI(
                name=_api_name,
                user_defined_callback=func,
                input_descriptor=input,
                output_descriptor=output,
                doc=api_doc,
                route=_route,
            )
            self.apis[api.name] = api

        return decorator
