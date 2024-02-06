from __future__ import annotations

import inspect
import io
import pathlib
import sys
import typing as t
from typing import ClassVar

from pydantic import BaseModel
from pydantic import Field
from pydantic import RootModel
from pydantic import create_model
from pydantic._internal._typing_extra import is_annotated
from typing_extensions import get_args

from bentoml._internal.service.openapi.specification import Schema

from .typing_utils import is_image_type
from .typing_utils import is_iterator_type
from .typing_utils import is_list_type
from .typing_utils import is_union_type
from .validators import ContentType

if t.TYPE_CHECKING:
    from starlette.requests import Request
    from starlette.responses import Response

    from _bentoml_impl.serde import Serde


DEFAULT_TEXT_MEDIA_TYPE = "text/plain"


def is_file_type(type_: type) -> bool:
    return issubclass(type_, pathlib.PurePath) or is_image_type(type_)


class IOMixin:
    multipart_fields: ClassVar[t.List[str]]
    media_type: ClassVar[t.Optional[str]] = None

    @classmethod
    def openapi_components(cls, name: str) -> dict[str, Schema]:
        from .service.openapi import REF_TEMPLATE

        if issubclass(cls, RootModel):
            return {}
        assert issubclass(cls, IOMixin) and issubclass(cls, BaseModel)
        json_schema = cls.model_json_schema(ref_template=REF_TEMPLATE)
        defs = json_schema.pop("$defs", None)
        main_name = (
            f"{name}__{cls.__name__}"
            if cls.__name__ in ("Input", "Output")
            else cls.__name__
        )
        json_schema["title"] = main_name
        components: dict[str, Schema] = {main_name: Schema(**json_schema)}
        if defs is not None:
            # NOTE: This is a nested models, hence we will update the definitions
            components.update({k: Schema(**v) for k, v in defs.items()})
        return components

    @classmethod
    def mime_type(cls) -> str:
        if cls.media_type is not None:
            return cls.media_type
        if not issubclass(cls, RootModel):
            if cls.multipart_fields:
                return "multipart/form-data"
            return "application/json"
        json_schema = cls.model_json_schema()
        if json_schema.get("type") == "string":
            return DEFAULT_TEXT_MEDIA_TYPE
        elif json_schema.get("type") == "file":
            if "content_type" in json_schema:
                return json_schema["content_type"]
            if (format := json_schema.get("format")) == "image":
                return "image/*"
            elif format == "audio":
                return "audio/*"
            elif format == "video":
                return "video/*"
            return "*/*"
        return "application/json"

    @classmethod
    def __pydantic_init_subclass__(cls) -> None:
        cls.multipart_fields = []
        for k, field in cls.model_fields.items():
            annotation = field.annotation
            try:
                if is_union_type(annotation):
                    any_of = get_args(annotation)
                    if len(any_of) != 2 or type(None) not in any_of:
                        raise TypeError("Union type is not supported yet")
                    annotation = next(a for a in any_of if a is not type(None))
                if is_list_type(annotation):
                    args = get_args(annotation)
                    annotation = args[0] if args else t.Any
                if is_annotated(annotation):
                    annotation = get_args(annotation)[0]
                if is_file_type(annotation):
                    cls.multipart_fields.append(k)
            except TypeError:
                pass

    @classmethod
    def from_inputs(cls, *args: t.Any, **kwargs: t.Any) -> IODescriptor:
        assert issubclass(cls, IODescriptor)
        model_fields = list(cls.model_fields)
        for i, arg in enumerate(args):
            if i < len(model_fields) and model_fields[i] == ARGS:
                kwargs[ARGS] = args[i:]
                break
            if i < len(model_fields):
                if model_fields[i] in kwargs:
                    raise TypeError(f"Duplicate arg given: {model_fields[i]}")
                kwargs[model_fields[i]] = arg
            else:
                raise TypeError("unexpected positional arg")
        extra_fields = set(kwargs.keys()) - set(cls.model_fields.keys())
        if KWARGS in model_fields:
            kwargs[KWARGS] = {k: kwargs.pop(k) for k in extra_fields}
        return cls(**kwargs)

    @classmethod
    async def from_http_request(cls, request: Request, serde: Serde) -> IODescriptor:
        """Parse a input model from HTTP request"""
        return await serde.parse_request(request, t.cast(t.Type[IODescriptor], cls))

    @classmethod
    async def to_http_response(cls, obj: t.Any, serde: Serde) -> Response:
        """Convert a output value to HTTP response"""
        import mimetypes

        from pydantic import RootModel
        from starlette.responses import FileResponse
        from starlette.responses import Response
        from starlette.responses import StreamingResponse

        from _bentoml_impl.serde import JSONSerde

        structured_media_type = cls.media_type or serde.media_type

        if inspect.isasyncgen(obj):

            async def async_stream() -> t.AsyncGenerator[str | bytes, None]:
                async for item in obj:
                    if isinstance(item, (str, bytes)):
                        yield item
                    else:
                        obj_item = cls(item) if issubclass(cls, RootModel) else item
                        yield serde.serialize_model(t.cast(IODescriptor, obj_item))

            return StreamingResponse(async_stream(), media_type=cls.mime_type())

        elif inspect.isgenerator(obj):

            def content_stream() -> t.Generator[str | bytes, None, None]:
                for item in obj:
                    if isinstance(item, (str, bytes)):
                        yield item
                    else:
                        obj_item = cls(item) if issubclass(cls, RootModel) else item
                        yield serde.serialize_model(t.cast(IODescriptor, obj_item))

            return StreamingResponse(content_stream(), media_type=cls.mime_type())
        elif not issubclass(cls, RootModel):
            if cls.multipart_fields:
                return Response(
                    "Multipart response is not supported yet", status_code=500
                )
            return Response(
                content=serde.serialize_model(t.cast(IODescriptor, obj)),
                media_type=structured_media_type,
            )
        else:
            if is_file_type(type(obj)) and isinstance(serde, JSONSerde):
                if isinstance(obj, pathlib.PurePath):
                    media_type = (
                        mimetypes.guess_type(obj)[0] or "application/octet-stream"
                    )
                    should_inline = media_type.startswith("image")
                    content_disposition_type = (
                        "inline" if should_inline else "attachment"
                    )
                    return FileResponse(
                        obj,
                        filename=obj.name,
                        media_type=media_type,
                        content_disposition_type=content_disposition_type,
                    )
                else:  # is PIL Image
                    buffer = io.BytesIO()
                    image_format = obj.format or "PNG"
                    obj.save(buffer, format=image_format)
                    return Response(
                        content=buffer.getvalue(),
                        media_type=f"image/{image_format.lower()}",
                    )

            if not isinstance(obj, RootModel):
                ins: IODescriptor = t.cast(IODescriptor, cls(obj))
            else:
                ins = t.cast(IODescriptor, obj)
            if isinstance(rendered := ins.model_dump(), (str, bytes)) and isinstance(
                serde, JSONSerde
            ):
                return Response(content=rendered, media_type=cls.mime_type())
            else:
                return Response(
                    content=serde.serialize_model(ins), media_type=structured_media_type
                )


ARGS = "args"
KWARGS = "kwargs"


class IODescriptor(IOMixin, BaseModel):
    @classmethod
    def from_input(
        cls, func: t.Callable[..., t.Any], *, skip_self: bool = False
    ) -> type[IODescriptor]:
        from pydantic._internal._typing_extra import eval_type_lenient

        try:
            module = sys.modules[func.__module__]
        except KeyError:
            global_ns = None
        else:
            global_ns = module.__dict__
        signature = inspect.signature(func)

        fields: dict[str, tuple[str, t.Any]] = {}
        parameter_tuples = iter(signature.parameters.items())
        if skip_self:
            next(parameter_tuples)

        for name, param in parameter_tuples:
            if name in ("context", "ctx"):
                # Reserved name for context object passed in
                continue
            annotation = param.annotation
            if annotation is param.empty:
                annotation = t.Any
            else:
                annotation = eval_type_lenient(annotation, global_ns, None)
            if param.kind == param.VAR_KEYWORD:
                name = KWARGS
                annotation = t.Dict[str, t.Any]
            elif param.kind == param.VAR_POSITIONAL:
                name = ARGS
                annotation = t.List[annotation]
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
        except (ValueError, TypeError) as e:
            raise TypeError(
                f"Unable to infer the input spec for function {func}, "
                "please specify input_spec manually"
            ) from e

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
            return_annotation = t.Any
        else:
            return_annotation = eval_type_lenient(
                signature.return_annotation, global_ns, None
            )
        media_type: str | None = None
        if is_iterator_type(return_annotation):
            return_annotation = get_args(return_annotation)[0]
        elif is_annotated(return_annotation):
            content_type = next(
                (a for a in get_args(return_annotation) if isinstance(a, ContentType)),
                None,
            )
            if content_type is not None:
                media_type = content_type.content_type
        try:
            output = ensure_io_descriptor(return_annotation)
            output.media_type = media_type
            return output
        except (ValueError, TypeError) as e:
            raise TypeError(
                f"Unable to infer the output spec for function {func}, "
                "please specify output_spec manually"
            ) from e


def ensure_io_descriptor(output_type: type) -> type[IODescriptor]:
    if inspect.isclass(output_type) and issubclass(output_type, BaseModel):
        if not issubclass(output_type, IODescriptor):

            class Output(IOMixin, output_type):
                pass

            return t.cast(t.Type[IODescriptor], Output)

        return output_type
    return t.cast(
        t.Type[IODescriptor],
        create_model("Output", __base__=(IOMixin, RootModel[output_type])),  # type: ignore
    )
