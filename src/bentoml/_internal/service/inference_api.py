from __future__ import annotations

import inspect
import re
import typing as t

import yaml
from pydantic import BaseModel
from pydantic import RootModel

from ...exceptions import InvalidArgument
from ..context import ServiceContext as Context
from ..io_descriptors import IODescriptor
from ..io_descriptors.base import IOType
from ..ionext.function import get_input_spec
from ..ionext.function import get_output_spec
from ..types import is_compatible_type

if t.TYPE_CHECKING:
    from starlette.requests import Request
    from starlette.responses import Response

RESERVED_API_NAMES = [
    "index",
    "swagger",
    "docs",
    "metrics",
    "healthz",
    "livez",
    "readyz",
]


class InferenceAPI(t.Generic[IOType]):
    def __init__(
        self,
        user_defined_callback: t.Callable[..., t.Any] | None,
        input_descriptor: IODescriptor[IOType],
        output_descriptor: IODescriptor[IOType],
        name: str | None,
        doc: str | None = None,
        route: str | None = None,
    ):
        if user_defined_callback is not None:
            # Use user_defined_callback function variable if name not specified
            name = user_defined_callback.__name__ if name is None else name
            # Use user_defined_callback function docstring `__doc__` if doc not specified
            doc = user_defined_callback.__doc__ if doc is None else doc
        else:
            name = "" if name is None else name
            doc = "" if doc is None else doc

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
                    and not is_compatible_type(input_type, annotation)
                ):
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
                        and not annotation == Context
                    ):
                        raise TypeError(
                            f"Expected type of argument '{second_arg}' to be 'bentoml.Context', got '{annotation}'"
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


def _InferenceAPI_dumper(dumper: yaml.Dumper, api: InferenceAPI[t.Any]) -> yaml.Node:
    return dumper.represent_dict(
        {
            "route": api.route,
            "doc": api.doc,
            "input": api.input.__class__.__name__,
            "output": api.output.__class__.__name__,
        }
    )


yaml.add_representer(InferenceAPI, _InferenceAPI_dumper)


class APIEndpoint:
    def __init__(
        self,
        user_defined_callback: t.Callable[..., t.Any],
        input_spec: type[BaseModel] | None = None,
        output_spec: type[BaseModel] | None = None,
        name: str | None = None,
        doc: str | None = None,
        route: str | None = None,
        mimetype: str | None = None,
        stream_output: bool | None = None,
    ):
        if name is None:
            name = user_defined_callback.__name__
        if doc is None:
            doc = user_defined_callback.__doc__
        if route is None:
            route = f"/{name}"
        InferenceAPI._validate_route(route)
        if input_spec is None:
            input_spec = get_input_spec(user_defined_callback)
            if input_spec is None:
                raise ValueError(
                    "The type annotation is not complete or supported for API inference"
                )
        if output_spec is None:
            output_spec = get_output_spec(user_defined_callback)
            if output_spec is None:
                raise ValueError(
                    "The type annotation is not complete or supported for API inference"
                )
        if stream_output is None:
            stream_output = inspect.isasyncgenfunction(
                user_defined_callback
            ) or inspect.isgeneratorfunction(user_defined_callback)

        parameters = inspect.signature(user_defined_callback).parameters
        self.ctx_param: str | None = None
        if "ctx" in parameters:
            self.ctx_param = "ctx"
        elif "context" in parameters:
            self.ctx_param = "context"

        self.name = name
        self.doc = doc
        self.route = route
        self.input_spec = input_spec
        self.output_spec = output_spec
        self.func = user_defined_callback
        self.stream_output = stream_output
        self.mimetype = mimetype or "application/json"

    def asdict(self) -> dict[str, t.Any]:
        return {
            "name": self.name,
            "route": self.route,
            "doc": self.doc,
            "input": self.input_spec.model_json_schema(),
            "output": self.output_spec.model_json_schema(),
            "stream_output": self.stream_output,
        }

    async def from_http_request(self, request: Request) -> BaseModel:
        # TODO: move this to input model
        json_str = await request.body()
        return self.input_spec.model_validate_json(json_str)

    async def to_http_response(self, obj: t.Any) -> Response:
        # TODO: move this to input model
        from starlette.responses import Response
        from starlette.responses import StreamingResponse

        output_spec = self.output_spec
        if not issubclass(output_spec, RootModel):
            return Response(content=obj.model_dump_json(), media_type=self.mimetype)
        if self.stream_output:
            if inspect.isasyncgen(obj):

                async def content_stream():
                    async for item in obj:
                        if isinstance(item, str):
                            yield item
                        else:
                            yield output_spec(item).model_dump_json()

            else:

                def content_stream():
                    for item in obj:
                        if isinstance(item, str):
                            yield item
                        else:
                            yield output_spec(item).model_dump_json()

            return StreamingResponse(content_stream(), media_type=self.mimetype)
        else:
            rendered = output_spec(obj).model_dump()
            if isinstance(rendered, (str, bytes)):
                return Response(content=rendered, media_type=self.mimetype)
            else:
                return Response(
                    content=output_spec(obj).model_dump_json(), media_type=self.mimetype
                )


def _APIEndpoint_dumper(dumper: yaml.Dumper, api: APIEndpoint) -> yaml.Node:
    return dumper.represent_dict(api.asdict())


yaml.add_representer(InferenceAPI, _APIEndpoint_dumper)
