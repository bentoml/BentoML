from __future__ import annotations

import functools
import inspect
import typing as t

import attrs

from bentoml._internal.service.openapi import SUCCESS_DESCRIPTION
from bentoml._internal.service.openapi.specification import MediaType
from bentoml._internal.service.openapi.specification import Schema
from bentoml._internal.utils import dict_filter_none

from .io_models import IODescriptor

R = t.TypeVar("R")
T = t.TypeVar("T", bound="APIMethod[..., t.Any]")
if t.TYPE_CHECKING:
    P = t.ParamSpec("P")
else:
    P = t.TypeVar("P")


DEFAULT_STREAM_MEDIA_TYPE = "text/event-stream"


def _only_include(data: dict[str, t.Any], fields: t.Container[str]) -> dict[str, t.Any]:
    return {k: v for k, v in data.items() if k in fields}


@attrs.define
class APIMethod(t.Generic[P, R]):
    func: t.Callable[t.Concatenate[t.Any, P], R]
    route: str = attrs.field()
    name: str = attrs.field()
    input_spec: type[IODescriptor] = attrs.field()
    output_spec: type[IODescriptor] = attrs.field()
    media_type: str | None = None
    batchable: bool = False
    batch_dim: tuple[int, int] = attrs.field(
        default=(0, 0), converter=lambda x: (x, x) if not isinstance(x, tuple) else x
    )
    max_batch_size: int = 100
    max_latency_ms: int = 60000
    is_stream: bool = attrs.field(init=False)
    doc: str | None = attrs.field(init=False)
    ctx_param: str | None = attrs.field(init=False)

    @doc.default
    def default_doc(self) -> str | None:
        return self.func.__doc__

    @ctx_param.default
    def default_ctx_param(self) -> str | None:
        parameters = inspect.signature(self.func).parameters
        if "ctx" in parameters:
            return "ctx"
        elif "context" in parameters:
            return "context"
        return None

    @is_stream.default
    def default_is_stream(self) -> bool:
        return inspect.isasyncgenfunction(self.func) or inspect.isgeneratorfunction(
            self.func
        )

    @name.default
    def default_name(self) -> str:
        return self.func.__name__

    @route.default
    def default_route(self) -> str:
        if self.func.__name__ == "__call__":
            return "/"
        return f"/{self.func.__name__}"

    @input_spec.default
    def default_input_spec(self) -> type[IODescriptor]:
        return IODescriptor.from_input(self.func, skip_self=True)

    @output_spec.default
    def default_output_spec(self) -> type[IODescriptor]:
        return IODescriptor.from_output(self.func)

    def __attrs_post_init__(self) -> None:
        if self.media_type:
            self.output_spec.media_type = self.media_type
        elif self.is_stream:
            self.output_spec.media_type = DEFAULT_STREAM_MEDIA_TYPE

    @t.overload
    def __get__(self: T, instance: None, owner: type) -> T:
        ...

    @t.overload
    def __get__(self, instance: object, owner: type) -> t.Callable[P, R]:
        ...

    def __get__(self: T, instance: t.Any, owner: type) -> t.Callable[P, R] | T:
        if instance is None:
            return self
        return t.cast("t.Callable[P, R]", functools.partial(self.func, instance))

    def schema(self) -> dict[str, t.Any]:
        output = _flatten_model_schema(self.output_spec)
        if self.is_stream:
            output["is_stream"] = True
        if self.output_spec.media_type:
            output["media_type"] = self.output_spec.media_type
        return dict_filter_none(
            {
                "name": self.name,
                "route": self.route,
                "description": self.__doc__,
                "batchable": self.batchable,
                "input": _flatten_model_schema(self.input_spec),
                "output": output,
            }
        )

    def openapi_request(self) -> dict[str, t.Any]:
        from .service.openapi import REF_TEMPLATE

        return {
            "content": {
                self.input_spec.mime_type(): MediaType(
                    schema=Schema(
                        **_only_include(
                            self.input_spec.model_json_schema(
                                ref_template=REF_TEMPLATE, mode="serialization"
                            ),
                            [attr.name for attr in Schema.__attrs_attrs__],
                        )
                    )
                )
            },
        }

    def openapi_response(self) -> dict[str, t.Any]:
        from .service.openapi import REF_TEMPLATE

        return {
            "description": SUCCESS_DESCRIPTION,
            "content": {
                self.output_spec.mime_type(): MediaType(
                    schema=Schema(
                        **_only_include(
                            self.output_spec.model_json_schema(
                                ref_template=REF_TEMPLATE, mode="serialization"
                            ),
                            [attr.name for attr in Schema.__attrs_attrs__],
                        )
                    )
                )
            },
        }


def _flatten_model_schema(model: type[IODescriptor]) -> dict[str, t.Any]:
    schema = model.model_json_schema()
    if not schema.get("properties") or "$defs" not in schema:
        return schema
    defs = schema.pop("$defs", {})
    for value in schema["properties"].values():
        if "allOf" in value:
            value.update(value.pop("allOf")[0])
        if "$ref" in value:
            ref = value.pop("$ref")[len("#/$defs/") :]
            value.update(defs[ref])
    return schema


@t.overload
def api(func: t.Callable[t.Concatenate[t.Any, P], R]) -> APIMethod[P, R]:
    ...


@t.overload
def api(
    *,
    route: str | None = ...,
    name: str | None = ...,
    input_spec: type[IODescriptor] | None = ...,
    output_spec: type[IODescriptor] | None = ...,
    media_type: str | None = ...,
    batchable: bool = ...,
    batch_dim: int | tuple[int, int] = ...,
    max_batch_size: int = ...,
    max_latency_ms: int = ...,
) -> t.Callable[[t.Callable[t.Concatenate[t.Any, P], R]], APIMethod[P, R]]:
    ...


def api(
    func: t.Callable[t.Concatenate[t.Any, P], R] | None = None,
    *,
    route: str | None = None,
    name: str | None = None,
    input_spec: type[IODescriptor] | None = None,
    output_spec: type[IODescriptor] | None = None,
    media_type: str | None = None,
    batchable: bool = False,
    batch_dim: int | tuple[int, int] = 0,
    max_batch_size: int = 100,
    max_latency_ms: int = 60000,
) -> (
    APIMethod[P, R]
    | t.Callable[[t.Callable[t.Concatenate[t.Any, P], R]], APIMethod[P, R]]
):
    def wrapper(func: t.Callable[t.Concatenate[t.Any, P], R]) -> APIMethod[P, R]:
        params: dict[str, t.Any] = {
            "media_type": media_type,
            "batchable": batchable,
            "batch_dim": batch_dim,
            "max_batch_size": max_batch_size,
            "max_latency_ms": max_latency_ms,
        }
        if route is not None:
            params["route"] = route
        if name is not None:
            params["name"] = name
        if input_spec is not None:
            params["input_spec"] = input_spec
        if output_spec is not None:
            params["output_spec"] = output_spec
        return APIMethod(func, **params)

    if func is not None:
        return wrapper(func)
    return wrapper
