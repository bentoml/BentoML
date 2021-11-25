

import typing

from starlette.requests import Request
from starlette.responses import Response
from starlette.routing import BaseRoute

class OpenAPIResponse(Response):
    media_type = ...
    def render(self, content: typing.Any) -> bytes: ...

class EndpointInfo(typing.NamedTuple):
    path: str
    http_method: str
    func: typing.Callable
    ...

class BaseSchemaGenerator:
    def get_schema(self, routes: typing.List[BaseRoute]) -> dict: ...
    def get_endpoints(
        self, routes: typing.List[BaseRoute]
    ) -> typing.List[EndpointInfo]:
        """
        Given the routes, yields the following information:

        - path
            eg: /users/
        - http_method
            one of 'get', 'post', 'put', 'patch', 'delete', 'options'
        - func
            method ready to extract the docstring
        """
        ...
    def parse_docstring(self, func_or_method: typing.Callable) -> dict:
        """
        Given a function, parse the docstring as YAML and return a dictionary of info.
        """
        ...
    def OpenAPIResponse(self, request: Request) -> Response: ...

class SchemaGenerator(BaseSchemaGenerator):
    def __init__(self, base_schema: dict) -> None: ...
    def get_schema(self, routes: typing.List[BaseRoute]) -> dict: ...
