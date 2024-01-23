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
        if self.is_stream and not self.output_spec.media_type:
            self.output_spec.media_type = DEFAULT_STREAM_MEDIA_TYPE

    @t.overload
    def __get__(self: T, instance: None, owner: type) -> T:
        ...

    @t.overload
    def __get__(self, instance: object, owner: type) -> t.Callable[P, R]:
        ...

    def __get__(self: T, instance: t.Any, owner: type) -> t.Callable[P, R] | T:
        from pydantic.fields import FieldInfo
        from pydantic_core import PydanticUndefined

        if instance is None:
            return self

        func_sig = inspect.signature(self.func)
        # skip the `self` parameter
        params = list(func_sig.parameters.values())[1:]

        @functools.wraps(self.func)
        def wrapped(*args: P.args, **kwargs: P.kwargs) -> R:
            # Extract the default values from `Field` if any
            for i, param in enumerate(params):
                if i < len(args) or param.name in kwargs:
                    # skip the arguments that are already provided
                    continue
                if isinstance(field := param.default, FieldInfo):
                    if field.default is not PydanticUndefined:
                        kwargs[param.name] = field.default
                    elif field.default_factory not in (None, PydanticUndefined):
                        kwargs[param.name] = field.default_factory()
            return self.func(instance, *args, **kwargs)

        wrapped.__signature__ = func_sig.replace(parameters=params)
        wrapped.__is_bentoml_api_func__ = True
        # same as functools.partial in order that inspect can recognize it
        wrapped.func = self.func
        return wrapped

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

        input = _flatten_field(
            _only_include(
                self.input_spec.model_json_schema(
                    ref_template=REF_TEMPLATE, mode="serialization"
                ),
                [attr.name for attr in Schema.__attrs_attrs__],
            ),
            {},
            max_depth=1,
        )

        return {
            "content": {self.input_spec.mime_type(): MediaType(schema=Schema(**input))},
        }

    def openapi_response(self) -> dict[str, t.Any]:
        from .service.openapi import REF_TEMPLATE

        output = _flatten_field(
            _only_include(
                self.output_spec.model_json_schema(
                    ref_template=REF_TEMPLATE, mode="serialization"
                ),
                [attr.name for attr in Schema.__attrs_attrs__],
            ),
            {},
            max_depth=1,
        )

        return {
            "description": SUCCESS_DESCRIPTION,
            "content": {
                self.output_spec.mime_type(): MediaType(schema=Schema(**output))
            },
        }


def _flatten_field(
    schema: dict[str, t.Any],
    defs: dict[str, t.Any],
    max_depth: int | None = None,
    _depth: int = 0,
) -> dict[str, t.Any]:
    if "allOf" in schema:
        schema.update(schema.pop("allOf")[0])
    if "anyOf" in schema:
        schema.update(schema.pop("anyOf")[0])
    if max_depth is not None and _depth >= max_depth:
        return schema
    if "$ref" in schema:
        ref = schema.pop("$ref")[len("#/$defs/") :]
        schema.update(_flatten_field(defs[ref], defs, max_depth, _depth=_depth + 1))
    elif schema.get("type") == "object" and "properties" in schema:
        for k, v in schema["properties"].items():
            schema["properties"][k] = _flatten_field(
                v, defs, max_depth, _depth=_depth + 1
            )
    elif schema.get("type") == "array" and "items" in schema:
        schema["items"] = _flatten_field(
            schema["items"], defs, max_depth, _depth=_depth + 1
        )
    return schema


def _flatten_model_schema(model: type[IODescriptor]) -> dict[str, t.Any]:
    schema = model.model_json_schema()
    if not schema.get("properties"):
        return schema
    defs = schema.pop("$defs", {})
    return _flatten_field(schema, defs)


class _SyncToAsyncDecorator(t.Protocol):
    @t.overload
    def __call__(
        self, func: t.Callable[P, t.Generator[R, None, None]]
    ) -> t.Callable[P, t.AsyncGenerator[R, None]]:
        ...

    @t.overload
    def __call__(
        self, func: t.Callable[P, R]
    ) -> t.Callable[P, t.Coroutine[None, None, R]]:
        ...


class _APIWrapper:
    @t.overload
    def __call__(self, func: t.Callable[t.Concatenate[t.Any, P], R]) -> APIMethod[P, R]:
        ...

    @t.overload
    def __call__(
        self,
        *,
        route: str | None = ...,
        name: str | None = ...,
        input_spec: type[IODescriptor] | None = ...,
        output_spec: type[IODescriptor] | None = ...,
        batchable: bool = ...,
        batch_dim: int | tuple[int, int] = ...,
        max_batch_size: int = ...,
        max_latency_ms: int = ...,
    ) -> t.Callable[[t.Callable[t.Concatenate[t.Any, P], R]], APIMethod[P, R]]:
        ...

    def __call__(
        self,
        func: t.Callable[t.Concatenate[t.Any, P], R] | None = None,
        *,
        route: str | None = None,
        name: str | None = None,
        input_spec: type[IODescriptor] | None = None,
        output_spec: type[IODescriptor] | None = None,
        batchable: bool = False,
        batch_dim: int | tuple[int, int] = 0,
        max_batch_size: int = 100,
        max_latency_ms: int = 60000,
    ) -> (
        APIMethod[P, R]
        | t.Callable[[t.Callable[t.Concatenate[t.Any, P], R]], APIMethod[P, R]]
    ):
        """Wrap a service method to be exposed as API endpoint.

        Args:
            func: The service method to be wrapped
            route: The route of the API endpoint, default to the name of the method prefixed with '/'
            name: The name of the API endpoint, default to the name of the method
            input_spec: The input spec of the API endpoint, will be derived if not provided
            output_spec: The output spec of the API endpoint, will be derived if not provided
            batchable: Whether the API endpoint is batchable
            batch_dim: The dimension of the batch, default to (0, 0)
            max_batch_size: The max batch size, default to 100
            max_latency_ms: The max latency in milliseconds, default to 60000

        Returns:
            The API method
        """

        def wrapper(func: t.Callable[t.Concatenate[t.Any, P], R]) -> APIMethod[P, R]:
            params: dict[str, t.Any] = {
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

    @t.overload
    def sync_to_async(
        self, func: t.Callable[P, t.Generator[R, None, None]]
    ) -> t.Callable[P, t.AsyncGenerator[R, None]]:
        ...

    @t.overload
    def sync_to_async(
        self, func: t.Callable[P, R]
    ) -> t.Callable[P, t.Coroutine[None, None, R]]:
        ...

    @t.overload
    def sync_to_async(self, *, threads: int = 1) -> _SyncToAsyncDecorator:
        ...

    def sync_to_async(
        self, func: t.Callable[..., t.Any] | None = None, *, threads: int = 1
    ) -> t.Callable[..., t.Any] | _SyncToAsyncDecorator:
        """Convert a sync function to async function

        Examples::

            @bentoml.api.sync_to_async
            def add(a: int, b: int) -> int:
                return a + b
            # to call: await add(1, 2)

            # Supports generators
            @bentoml.api.sync_to_async
            def my_generator(end: int) -> Generator[int, None, None]:
                for i in range(end):
                    yield i
            # to call: async for i in my_generator(3): pass
        """

        import anyio
        import anyio.to_thread

        def wrapper(func: t.Callable[P, t.Any]) -> t.Callable[P, t.Any]:
            async def wrapped(*args: P.args, **kwargs: P.kwargs) -> t.Any:
                return await anyio.to_thread.run_sync(
                    functools.partial(func, **kwargs),
                    *args,
                    limiter=anyio.CapacityLimiter(threads),
                )

            async def wrapped_generator(
                *args: P.args, **kwargs: P.kwargs
            ) -> t.AsyncGenerator[t.Any, None]:
                gen = func(*args, **kwargs)
                while True:
                    try:
                        yield await anyio.to_thread.run_sync(
                            gen.__next__, limiter=anyio.CapacityLimiter(threads)
                        )
                    except StopIteration:
                        break
                    except RuntimeError as e:
                        if "raised StopIteration" in str(e):
                            break
                        raise

            if inspect.isgeneratorfunction(func):
                new_func = wrapped_generator
            else:
                new_func = wrapped
            new_func.__signature__ = inspect.signature(func)
            return functools.update_wrapper(new_func, func)

        if func is not None:
            return wrapper(func)
        return wrapper


api = _APIWrapper()
