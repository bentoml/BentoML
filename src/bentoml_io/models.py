from __future__ import annotations

import inspect
import sys
import typing as t

from pydantic import BaseModel
from pydantic import Field
from pydantic import RootModel
from pydantic import create_model
from typing_extensions import get_args

from .typing_utils import is_iterator_type

if t.TYPE_CHECKING:
    from starlette.requests import Request
    from starlette.responses import Response

    from .serde import Serde


class IOMixin:
    @classmethod
    async def from_http_request(cls, request: Request, serde: Serde) -> BaseModel:
        """Parse a input model from HTTP request"""
        json_str = await request.body()
        return serde.deserialize_model(json_str, t.cast(t.Type[BaseModel], cls))

    @classmethod
    async def to_http_response(cls, obj: t.Any, serde: Serde) -> Response:
        """Convert a output value to HTTP response"""
        from pydantic import RootModel
        from starlette.responses import Response
        from starlette.responses import StreamingResponse

        if not issubclass(cls, RootModel):
            return Response(
                content=serde.serialize_model(t.cast("BaseModel", obj)),
                media_type=serde.media_type,
            )
        if inspect.isasyncgen(obj):

            async def async_stream() -> t.AsyncGenerator[str | bytes, None]:
                async for item in obj:
                    if isinstance(item, (str, bytes)):
                        yield item
                    else:
                        yield serde.serialize_model(cls(item))

            return StreamingResponse(async_stream(), media_type="text/plain")

        elif inspect.isgenerator(obj):

            def content_stream() -> t.Generator[str | bytes, None, None]:
                for item in obj:
                    if isinstance(item, (str, bytes)):
                        yield item
                    else:
                        yield serde.serialize_model(cls(item))

            return StreamingResponse(content_stream(), media_type="text/plain")
        else:
            if not isinstance(obj, RootModel):
                ins = cls(obj)
            else:
                ins = obj
            if isinstance(rendered := ins.model_dump(), (str, bytes)):
                media_type = cls.model_json_schema().get("media_type", "text/plain")
                return Response(content=rendered, media_type=media_type)
            else:
                return Response(
                    content=serde.serialize_model(ins), media_type=serde.media_type
                )


class IODescriptor(BaseModel, IOMixin):
    @classmethod
    def from_input(
        cls, func: t.Callable[..., t.Any], *, skip_self: bool = False
    ) -> type[IODescriptor]:
        signature = inspect.signature(func)

        fields: dict[str, tuple[str, t.Any]] = {}
        parameter_tuples = iter(signature.parameters.items())
        if skip_self:
            next(parameter_tuples)

        for name, param in parameter_tuples:
            if name in ("context", "ctx"):
                # Reserved name for context object passed in
                continue
            if param.kind in (param.VAR_KEYWORD, param.VAR_POSITIONAL):
                raise TypeError(
                    f"Unable to infer the input spec for function {func} because of var args, "
                    "please specify input_spec manually"
                ) from None
            annotation = param.annotation
            if annotation is param.empty:
                raise TypeError(
                    f"Missing type annotation for parameter {name} in function {func}"
                )
            default = param.default
            if default is param.empty:
                default = Field()
            fields[name] = (annotation, default)

        try:
            return t.cast(
                t.Type[IODescriptor],
                create_model(
                    "Input", __module__=func.__module__, __base__=IODescriptor, **fields
                ),  # type: ignore
            )
        except (ValueError, TypeError):
            raise TypeError(
                f"Unable to infer the input spec for function {func}, "
                "please specify input_spec manually"
            ) from None

    @classmethod
    def from_output(cls, func: t.Callable[..., t.Any]) -> type[IODescriptor]:
        from pydantic._internal._typing_extra import eval_type_lenient

        try:
            module = sys.modules[func.__module__]
        except KeyError:
            global_ns = None
        else:
            global_ns = module.__dict__
        signature = inspect.signature(func)
        if signature.return_annotation is inspect.Signature.empty:
            raise TypeError(f"Missing return type annotation for function {func}")
        return_annotation = eval_type_lenient(
            signature.return_annotation, global_ns, None
        )

        if is_iterator_type(return_annotation):
            return_annotation = get_args(return_annotation)[0]
        try:
            return ensure_io_descriptor(return_annotation)
        except (ValueError, TypeError):
            raise TypeError(
                f"Unable to infer the output spec for function {func}, "
                "please specify output_spec manually"
            ) from None


def ensure_io_descriptor(output_type: type) -> type[IODescriptor]:
    if issubclass(output_type, BaseModel):
        if not issubclass(output_type, IODescriptor):

            class Output(output_type, IOMixin):
                pass

            return t.cast(t.Type[IODescriptor], Output)

        return output_type
    return t.cast(
        t.Type[IODescriptor],
        create_model("Output", __base__=(IOMixin, RootModel[output_type])),  # type: ignore
    )
