from __future__ import annotations

import typing as t

from .emitter import Emitter
from .representer import BaseRepresenter
from .representer import Representer
from .representer import SafeRepresenter
from .resolver import BaseResolver
from .resolver import Resolver
from .serializer import Serializer

class BaseDumper(Emitter, Serializer, BaseRepresenter, BaseResolver):
    def __init__(
        self,
        stream: t.IO[str],
        default_style: str = ...,
        default_flow_style: bool = ...,
        canonical: bool | None = ...,
        indent: int = ...,
        width: int = ...,
        allow_unicode: bool | None = ...,
        line_break: str = ...,
        encoding: str = ...,
        explicit_start: bool | None = ...,
        explicit_end: bool | None = ...,
        version: bool | None = ...,
        tags: bool | None = ...,
        sort_keys: bool = ...,
    ) -> None: ...

class SafeDumper(Emitter, Serializer, SafeRepresenter, Resolver):
    def __init__(
        self,
        stream: t.IO[str],
        default_style: str = ...,
        default_flow_style: bool = ...,
        canonical: bool | None = ...,
        indent: int = ...,
        width: int = ...,
        allow_unicode: bool | None = ...,
        line_break: str = ...,
        encoding: str = ...,
        explicit_start: bool | None = ...,
        explicit_end: bool | None = ...,
        version: bool | None = ...,
        tags: bool | None = ...,
        sort_keys: bool = ...,
    ) -> None: ...

class Dumper(Emitter, Serializer, Representer, Resolver):
    def __init__(
        self,
        stream: t.IO[str],
        default_style: str = ...,
        default_flow_style: bool = ...,
        canonical: bool | None = ...,
        indent: int = ...,
        width: int = ...,
        allow_unicode: bool | None = ...,
        line_break: str = ...,
        encoding: str = ...,
        explicit_start: bool | None = ...,
        explicit_end: bool | None = ...,
        version: bool | None = ...,
        tags: bool | None = ...,
        sort_keys: bool = ...,
    ) -> None: ...
