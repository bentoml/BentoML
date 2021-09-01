import re
from typing import Optional

from ...exceptions import InvalidArgument
from ..io_descriptors import IODescriptor

RESERVED_API_NAMES = [
    "index",
    "swagger",
    "docs",
    "healthz",
    "metrics",
    "feedback",
]


class InferenceAPI:
    def __init__(
        self,
        name: str,
        user_defined_callback: callable,
        input_descriptor: IODescriptor,
        output_descriptor: IODescriptor,
        doc: Optional[str] = None,
        route: Optional[str] = None,
    ):
        self._validate_name(name)
        self._validate_route(route)

        self.name = name
        self.func = user_defined_callback
        self.input = input_descriptor
        self.output = output_descriptor
        self.doc = doc
        self.route = route

    @staticmethod
    def _validate_name(api_name: str):
        if not api_name.isidentifier():
            raise InvalidArgument(
                "Invalid API name: '{}', a valid identifier may only contain letters,"
                " numbers, underscores and not starting with a number.".format(api_name)
            )

        if api_name in RESERVED_API_NAMES:
            raise InvalidArgument(
                "Reserved API name: '{}' is reserved for infra endpoints".format(
                    api_name
                )
            )

    @staticmethod
    def _validate_route(route: str):
        if re.findall(
            r"[?#]+|^(//)|^:", route
        ):  # contains '?' or '#' OR  start with '//' OR start with ':'
            # https://tools.ietf.org/html/rfc3986#page-22
            raise InvalidArgument(
                "The path {} contains illegal url characters".format(route)
            )
        if route in RESERVED_API_NAMES:
            raise InvalidArgument(
                "Reserved API route: '{}' is reserved for infra endpoints".format(route)
            )
