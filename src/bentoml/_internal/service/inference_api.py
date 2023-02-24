from __future__ import annotations

import re
import typing as t
import inspect
from typing import Optional

import yaml

from ..types import is_compatible_type
from ..context import InferenceApiContext as Context
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
        user_defined_callback: t.Callable[..., t.Any] | None,
        input_descriptor: IODescriptor[t.Any],
        output_descriptor: IODescriptor[t.Any],
        name: Optional[str],
        doc: Optional[str] = None,
        route: Optional[str] = None,
    ):
        if user_defined_callback is not None:
            # Use user_defined_callback function variable if name not specified
            name = user_defined_callback.__name__ if name is None else name
            # Use user_defined_callback function docstring `__doc__` if doc not specified
            doc = user_defined_callback.__doc__ if doc is None else doc
        else:
            name, doc = "", ""

        # Use API name as route if route not specified
        route = name if route is None else route

        self.needs_ctx = False
        self.ctx_param = None
        input_type = input_descriptor.input_type()

        if user_defined_callback is not None:
            InferenceAPI._validate_name(name)
            InferenceAPI._validate_route(route)

            sig = inspect.signature(user_defined_callback)

            if len(sig.parameters) == 0:
                raise ValueError("Expected API function to take parameters.")

            if isinstance(input_type, dict):
                # note: in python 3.6 kwarg order was not guaranteed to be preserved,
                #       though it is in practice.
                for key in sig.parameters:
                    if key not in input_type:
                        if (
                            key in ["context", "ctx"]
                            or sig.parameters[key].annotation == Context
                        ):
                            if self.needs_ctx:
                                raise ValueError(
                                    f"API function has two context parameters: '{self.ctx_param}' and '{key}'; it should only have one."
                                )

                            self.needs_ctx = True
                            self.ctx_param = key
                            continue

                        raise ValueError(
                            f"API function has extra parameter with name '{key}'."
                        )

                    annotation: t.Type[t.Any] = sig.parameters[key].annotation
                    if (
                        isinstance(annotation, t.Type)
                        and annotation != inspect.Signature.empty
                    ):
                        # if type annotations have been successfully resolved
                        if not is_compatible_type(input_type[key], annotation):
                            raise TypeError(
                                f"Expected type of argument '{key}' to be '{input_type[key]}', got '{sig.parameters[key].annotation}'"
                            )

                expected_args = len(input_type) + (1 if self.needs_ctx else 0)
                if len(sig.parameters) != expected_args:
                    raise ValueError(
                        f"expected API function to have arguments ({', '.join(input_type.keys())}, [context]), got ({', '.join(sig.parameters.keys())})"
                    )

            else:
                param_iter = iter(sig.parameters)
                first_arg = next(param_iter)
                annotation = sig.parameters[first_arg].annotation
                if (
                    isinstance(annotation, t.Type)
                    and annotation != inspect.Signature.empty
                ):
                    if not is_compatible_type(input_type, annotation):
                        raise TypeError(
                            f"Expected type of argument '{first_arg}' to be '{input_type}', got '{sig.parameters[first_arg].annotation}'"
                        )

                if len(sig.parameters) > 2:
                    raise ValueError(
                        "API function should only take one or two arguments"
                    )
                elif len(sig.parameters) == 2:
                    self.needs_ctx = True

                    second_arg = next(param_iter)
                    annotation = sig.parameters[second_arg].annotation
                    if (
                        isinstance(annotation, t.Type)
                        and annotation != inspect.Signature.empty
                    ):
                        if not annotation == Context:
                            raise TypeError(
                                f"Expected type of argument '{second_arg}' to be '{input_type}', got '{sig.parameters[second_arg].annotation}'"
                            )

        if user_defined_callback is not None:
            self.func = user_defined_callback
        else:

            def nop(*args: t.Any, **kwargs: t.Any):
                return

            self.func = nop

        self.name = name
        self.multi_input = isinstance(input_type, dict)
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
