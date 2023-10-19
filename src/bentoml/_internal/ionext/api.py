from __future__ import annotations

import functools
import inspect
import typing as t

import attrs

from .function import get_input_spec
from .function import get_output_spec
from .models import IODescriptor

R = t.TypeVar("R")
T = t.TypeVar("T", bound="APIMethod[..., t.Any]")
if t.TYPE_CHECKING:
    P = t.ParamSpec("P")
else:
    P = t.TypeVar("P")


@attrs.define
class APIMethod(t.Generic[P, R]):
    func: t.Callable[t.Concatenate[t.Any, P], R]
    route: str = attrs.field()
    name: str = attrs.field()
    input_spec: type[IODescriptor] = attrs.field()
    output_spec: type[IODescriptor] = attrs.field()
    batchable: bool = False
    doc: str | None = attrs.field(init=False)

    @doc.default
    def default_doc(self) -> str | None:
        return self.func.__doc__

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
        spec = get_input_spec(self.func, skip_self=True)
        if spec is None:
            raise TypeError(
                f"Unable to infer the input spec for function {self.func}, "
                "please specify input_spec manually"
            )
        return spec

    @output_spec.default
    def default_output_spec(self) -> type[IODescriptor]:
        spec = get_output_spec(self.func)
        if spec is None:
            raise TypeError(
                f"Unable to infer the output spec for function {self.func}, "
                "please specify output_spec manually"
            )
        return spec

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
        output = self.output_spec.model_json_schema()
        if inspect.isasyncgenfunction(self.func) or inspect.isgeneratorfunction(
            self.func
        ):
            output["is_stream"] = True
        return {
            "name": self.name,
            "route": self.route,
            "doc": self.__doc__,
            "batchable": self.batchable,
            "input": self.input_spec.model_json_schema(),
            "output": output,
        }


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
    batchable: bool = ...,
) -> t.Callable[[t.Callable[t.Concatenate[t.Any, P], R]], APIMethod[P, R]]:
    ...


def api(
    func: t.Callable[t.Concatenate[t.Any, P], R] | None = None,
    *,
    route: str | None = None,
    name: str | None = None,
    input_spec: type[IODescriptor] | None = None,
    output_spec: type[IODescriptor] | None = None,
    batchable: bool = False,
) -> (
    APIMethod[P, R]
    | t.Callable[[t.Callable[t.Concatenate[t.Any, P], R]], APIMethod[P, R]]
):
    def wrapper(func: t.Callable[t.Concatenate[t.Any, P], R]) -> APIMethod[P, R]:
        params: dict[str, t.Any] = {"batchable": batchable}
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
