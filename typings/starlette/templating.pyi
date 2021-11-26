import typing
from starlette.background import BackgroundTask
from starlette.responses import Response
from starlette.types import Receive, Scope, Send

class _TemplateResponse(Response):
    media_type = ...
    def __init__(
        self,
        template: typing.Any,
        context: dict,
        status_code: int = ...,
        headers: dict = ...,
        media_type: str = ...,
        background: BackgroundTask = ...,
    ) -> None: ...
    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None: ...

class Jinja2Templates:
    def __init__(self, directory: str) -> None: ...
    def get_template(self, name: str) -> jinja2.Template: ...
    def TemplateResponse(
        self,
        name: str,
        context: dict,
        status_code: int = ...,
        headers: dict = ...,
        media_type: str = ...,
        background: BackgroundTask = ...,
    ) -> _TemplateResponse: ...
