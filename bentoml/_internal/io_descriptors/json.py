import typing as t

from bentoml.exceptions import InvalidArgument

from ..types import HTTPRequest, HTTPResponse
from ..utils import LazyLoader, catch_exceptions
from .base import IODescriptor

if t.TYPE_CHECKING:
    import pydantic
else:
    pydantic = LazyLoader("pydantic", globals(), "pydantic")


class JSON(IODescriptor):
    def __init__(self, pydantic_model: "pydantic.main.BaseModel" = None):
        if pydantic_model is not None:
            if not isinstance(pydantic_model, pydantic.BaseModel):
                raise InvalidArgument(
                    "Invalid argument type 'pydantic_model' for JSON io descriptor,"
                    "must be an instance of a pydantic model"
                )
            self._pydantic_model = pydantic_model

    def openapi_request_schema(self):
        pass

    def openapi_responses_schema(self):
        pass

    def from_http_request(self, request: HTTPRequest) -> str:
        pass

    def to_http_response(self, obj: str) -> HTTPResponse:
        pass
