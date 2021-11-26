import re
import typing as t
from typing import Optional

import yaml

from ...exceptions import InvalidArgument
from ..io_descriptors import IODescriptor

RESERVED_API_NAMES = [
    "index",
    "swagger",
    "docs",
    "metrics",
    "healthz",
    "livez",
    "readyz",
]


class InferenceAPI:
    def __init__(
        self,
        user_defined_callback: t.Callable[..., t.Any],
        input_descriptor: IODescriptor[t.Any],
        output_descriptor: IODescriptor[t.Any],
        name: Optional[str],
        doc: Optional[str] = None,
        route: Optional[str] = None,
    ):
        # Use user_defined_callback function variable if name not specified
        name = user_defined_callback.__name__ if name is None else name
        # Use user_defined_callback function docstring `__doc__` if doc not specified
        doc = user_defined_callback.__doc__ if doc is None else doc
        # Use API name as route if route not specified
        route = name if route is None else route

        InferenceAPI._validate_name(name)
        InferenceAPI._validate_route(route)

        self.name = name
        self.func = user_defined_callback
        self.input = input_descriptor
        self.output = output_descriptor
        self.doc = doc
        self.route = route

    def __str__(self):
        return f"{self.__class__.__name__}({str(self.input)} → {str(self.output)})"

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


def _InferenceAPI_dumper(dumper: yaml.Dumper, api: InferenceAPI) -> yaml.Node:
    return dumper.represent_dict(
        {
            "route": api.route,
            "doc": api.doc,
            "input": api.input.__class__.__name__,
            "output": api.output.__class__.__name__,
        }
    )


yaml.add_representer(InferenceAPI, _InferenceAPI_dumper)
