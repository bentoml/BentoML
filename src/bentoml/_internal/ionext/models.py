from __future__ import annotations

import inspect
import typing as t

from pydantic import BaseModel

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
    pass
