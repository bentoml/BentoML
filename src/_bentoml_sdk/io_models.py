from __future__ import annotations

import inspect
import io
import json
import logging
import pathlib
import sys
import typing as t
from typing import ClassVar

from pydantic import BaseModel
from pydantic import Field
from pydantic import TypeAdapter
from pydantic import create_model
from pydantic._internal._typing_extra import is_annotated
from starlette.responses import Response
from typing_extensions import get_args

from bentoml._internal.service.openapi.specification import Schema

from .typing_utils import is_image_type
from .typing_utils import is_iterator_type
from .typing_utils import is_list_type
from .typing_utils import is_union_type
from .validators import ContentType

if t.TYPE_CHECKING:
    from starlette.background import BackgroundTask
    from starlette.requests import Request
    from starlette.types import Receive
    from starlette.types import Scope
    from starlette.types import Send

    from _bentoml_impl.serde import Serde


try:
    from pydantic._internal._typing_extra import eval_type_lenient
except ImportError:
    from pydantic._internal._typing_extra import try_eval_type

    def eval_type_lenient(
        value: t.Any,
        globalns: dict[str, t.Any] | None = None,
        localns: dict[str, t.Any] | None = None,
    ) -> t.Any:
        ev, _ = try_eval_type(value, globalns, localns)
        return ev


DEFAULT_TEXT_MEDIA_TYPE = "text/plain"
logger = logging.getLogger("bentoml.serve")


def is_file_type(type_: type) -> bool:
    return issubclass(type_, pathlib.PurePath) or is_image_type(type_)


class IterableResponse(Response):
    def __init__(
        self,
        content: t.Iterable[memoryview | bytes],
        headers: t.Optional[t.Mapping[str, str]] = None,
        media_type: str | None = None,
        status_code: int = 200,
        background: t.Optional[BackgroundTask] = None,
    ) -> None:
        self.status_code = status_code
        if media_type is not None:
            self.media_type = media_type
        self.background = background
        self.body_iterator = iter(content)
        self.init_headers(headers)

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        await send(
            {
                "type": "http.response.start",
                "status": self.status_code,
                "headers": self.raw_headers,
            }
        )
        for chunk in self.body_iterator:
            await send({"type": "http.response.body", "body": chunk, "more_body": True})
        await send({"type": "http.response.body", "body": b"", "more_body": False})

        if self.background is not None:
            await self.background()


class IOMixin:
    multipart_fields: ClassVar[t.List[str]]
    media_type: ClassVar[t.Optional[str]] = None

    @classmethod
    def model_json_schema(cls, *args: t.Any, **kwargs: t.Any) -> dict[str, t.Any]:
        schema = super().model_json_schema(*args, **kwargs)
        if getattr(cls, "__root_input__", False):
            schema["root_input"] = True
        return schema

    @classmethod
    def openapi_components(cls, name: str) -> dict[str, Schema]:
        from .service.openapi import REF_TEMPLATE

        assert issubclass(cls, BaseModel)
        json_schema = cls.model_json_schema(ref_template=REF_TEMPLATE)
        defs = json_schema.pop("$defs", None)
        components: dict[str, Schema] = {}
        if not issubclass(cls, IORootModel):
            main_name = (
                f"{name}__{cls.__name__}"
                if cls.__name__ in ("Input", "Output")
                else cls.__name__
            )
            json_schema["title"] = main_name
            components[main_name] = Schema(**json_schema)
        if defs is not None:
            # NOTE: This is a nested models, hence we will update the definitions
            components.update({k: Schema(**v) for k, v in defs.items()})
        return components

    @classmethod
    def mime_type(cls) -> str:
        if cls.media_type is not None:
            return cls.media_type
        if not issubclass(cls, IORootModel):
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
            return "application/octet-stream"
        return "application/json"

    @classmethod
    def __get_pydantic_core_schema__(cls: type[BaseModel], source, handler):
        from ._pydantic import patch_annotation

        for _, info in cls.model_fields.items():
            ann, metadata = patch_annotation(
                info.annotation, cls.model_config, info.metadata
            )
            info.annotation = ann
            info.metadata = metadata

        return super().__get_pydantic_core_schema__(source, handler)

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
        if getattr(cls, "__root_input__", False):
            if len(args) > 1:
                raise TypeError("Expected exactly 1 argument")
            if len(args) < 1:
                return cls()
            arg = args[0]
            if issubclass(cls, IORootModel) or not isinstance(arg, dict):
                return cls(arg)
            else:
                return cls(**arg)
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
        import itertools
        import mimetypes

        from starlette.responses import FileResponse
        from starlette.responses import Response
        from starlette.responses import StreamingResponse

        if inspect.isasyncgen(obj):
            try:
                # try if there is any error before the first yield
                # if so, the error can be surfaced in the response
                first_chunk = await obj.__anext__()
            except StopAsyncIteration:
                pre_chunks = []
            else:
                pre_chunks = [first_chunk]

            async def gen() -> t.AsyncGenerator[t.Any, None]:
                for chunk in pre_chunks:
                    yield chunk
                async for chunk in obj:
                    yield chunk

            async def async_stream() -> t.AsyncGenerator[str | bytes, None]:
                try:
                    async for item in gen():
                        if isinstance(item, (str, bytes)):
                            yield item
                        else:
                            obj_item = (
                                cls(item) if issubclass(cls, IORootModel) else item
                            )
                            for chunk in serde.serialize_model(
                                t.cast(IODescriptor, obj_item)
                            ).data:
                                yield chunk
                except Exception:
                    logger.exception("Error while streaming response")

            return StreamingResponse(async_stream(), media_type=cls.mime_type())

        elif inspect.isgenerator(obj):
            trial, obj = itertools.tee(obj)
            try:
                next(trial)  # try if there is any error before the first yield
            except StopIteration:
                pass

            def content_stream() -> t.Generator[str | bytes, None, None]:
                try:
                    for item in obj:
                        if isinstance(item, (str, bytes)):
                            yield item
                        else:
                            obj_item = (
                                cls(item) if issubclass(cls, IORootModel) else item
                            )
                            yield from serde.serialize_model(
                                t.cast(IODescriptor, obj_item)
                            ).data
                except Exception:
                    logger.exception("Error while streaming response")

            return StreamingResponse(content_stream(), media_type=cls.mime_type())
        elif not issubclass(cls, IORootModel):
            if cls.multipart_fields:
                return Response(
                    "Multipart response is not supported yet", status_code=500
                )
            payload = serde.serialize_model(t.cast(IODescriptor, obj))
            return IterableResponse(
                content=payload.data,
                media_type=cls.mime_type(),
                headers=payload.headers,
            )
        else:
            if is_file_type(type(obj)):
                if isinstance(obj, pathlib.PurePath):
                    media_type = mimetypes.guess_type(obj)[0] or cls.mime_type()
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

            if not isinstance(obj, IORootModel):
                ins: IODescriptor = t.cast(IODescriptor, cls(obj))
            else:
                ins = t.cast(IODescriptor, obj)
            if isinstance(rendered := ins.model_dump(), (str, bytes)):
                return Response(content=rendered, media_type=cls.mime_type())
            else:
                payload = serde.serialize_model(ins)
                return IterableResponse(
                    content=payload.data,
                    media_type=cls.mime_type(),
                    headers=payload.headers,
                )


ARGS = "args"
KWARGS = "kwargs"


class IODescriptor(IOMixin, BaseModel):
    @classmethod
    def from_input(
        cls,
        func: t.Callable[..., t.Any],
        *,
        skip_self: bool = False,
        skip_names: t.Container[str] = (),
    ) -> type[IODescriptor]:
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
        positional_only_param: t.Any = None
        positional_only_default: t.Any = signature.empty
        for name, param in parameter_tuples:
            if name in skip_names:
                # Reserved name for context object passed in
                continue
            annotation = param.annotation
            if annotation is param.empty:
                annotation = t.Any
            else:
                annotation = eval_type_lenient(annotation, global_ns, None)
            if param.kind == param.POSITIONAL_ONLY:
                if positional_only_param is None:
                    positional_only_param = annotation
                    positional_only_default = param.default
                    continue

            if positional_only_param is not None:
                raise TypeError(
                    "When positional-only argument is used, no other parameters can be specified"
                )
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
            if positional_only_param is not None:
                content_type = None
                if is_annotated(positional_only_param):
                    content_type = next(
                        (
                            a
                            for a in get_args(positional_only_param)
                            if isinstance(a, ContentType)
                        ),
                        None,
                    )
                typ_ = ensure_io_descriptor(
                    positional_only_param, positional_only_default
                )
                typ_.__root_input__ = True
                typ_.media_type = content_type.content_type if content_type else None
                return typ_
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


RootModelRootType = t.TypeVar("RootModelRootType")


class IORootModel(IODescriptor, t.Generic[RootModelRootType]):
    root: RootModelRootType

    def __init__(self, root: RootModelRootType) -> None:
        super().__init__(root=root)  # type: ignore

    def model_dump(self, **kwargs: t.Any) -> dict[str, t.Any]:
        return super().model_dump(**kwargs)["root"]

    def model_dump_json(self, **kwargs: t.Any) -> str:
        return json.dumps(self.model_dump(mode="json", **kwargs))

    @classmethod
    def model_validate(cls, obj: t.Any, **kwargs: t.Any) -> t.Self:
        return super().model_validate({"root": obj}, **kwargs)

    @classmethod
    def model_validate_json(
        cls, json_data: str | bytes | bytearray, **kwargs: t.Any
    ) -> t.Self:
        json_schema = cls.model_json_schema()
        if json_schema.get("type") in ("string", "file"):
            parsed = json_data
        else:
            parsed = json.loads(json_data)
        return cls.model_validate(parsed, **kwargs)

    @classmethod
    def model_json_schema(cls, *args: t.Any, **kwargs: t.Any) -> dict[str, t.Any]:
        field = cls.model_fields["root"]
        if field.metadata:
            typ_ = t.Annotated[(field.annotation, *field.metadata)]
        else:
            typ_ = field.annotation
        schema = TypeAdapter(typ_).json_schema(*args, **kwargs)
        if getattr(cls, "__root_input__", False):
            schema["root_input"] = True
        return schema


def ensure_io_descriptor(
    typ_: type, root_default: t.Any = inspect.Signature.empty
) -> type[IODescriptor]:
    from pydantic._internal._utils import lenient_issubclass

    type_name = getattr(typ_, "__name__", "")

    if inspect.isclass(typ_) and lenient_issubclass(typ_, BaseModel):
        if not issubclass(typ_, IODescriptor):
            return t.cast(
                t.Type[IODescriptor],
                create_model(f"{type_name}IODescriptor", __base__=(IOMixin, typ_)),
            )
        return typ_

    if root_default is inspect.Signature.empty:
        extras = {}
    else:
        extras = {"root": (typ_, root_default)}

    return t.cast(
        t.Type[IODescriptor],
        create_model(f"{type_name}IODescriptor", __base__=IORootModel[typ_], **extras),
    )
