from starlette.requests import Request
from starlette.responses import Response

from .base import IODescriptor


class Text(IODescriptor):
    def openapi_request_schema(self):
        pass

    def openapi_responses_schema(self):
        pass

    async def from_http_request(self, request: Request) -> str:
        pass

    async def to_http_response(self, obj: str) -> Response:
        pass
