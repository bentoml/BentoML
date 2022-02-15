from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import typing as t

    from starlette.types import Send as ASGISend
    from starlette.types import Scope as ASGIScope
    from starlette.types import ASGIApp
    from starlette.types import Message as ASGIMessage
    from starlette.types import Receive as ASGIReceive

    class AsgiMiddleware(t.Protocol):
        def __call__(self, app: ASGIApp, **options: t.Any) -> ASGIApp:
            ...

    __all__ = [
        "AsgiMiddleware",
        "ASGIApp",
        "ASGIScope",
        "ASGISend",
        "ASGIReceive",
        "ASGIMessage",
    ]
