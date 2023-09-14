from typing import IO
from typing import Any

from .error import YAMLError

class EmitterError(YAMLError): ...

class Emitter:
    DEFAULT_TAG_PREFIXES: Any
    stream: Any
    encoding: Any
    states: Any
    state: Any
    events: Any
    event: Any
    indents: Any
    indent: Any
    flow_level: Any
    root_context: Any
    sequence_context: Any
    mapping_context: Any
    simple_key_context: Any
    line: Any
    column: Any
    whitespace: Any
    indention: Any
    open_ended: Any
    canonical: Any
    allow_unicode: Any
    best_indent: Any
    best_width: Any
    best_line_break: Any
    tag_prefixes: Any
    prepared_anchor: Any
    prepared_tag: Any
    analysis: Any
    style: Any

    def __init__(
        self,
        stream: IO[str],
        canonical: bool | None = ...,
        indent: int = ...,
        width: int = ...,
        allow_unicode: bool | None = ...,
        line_break: str = ...,
    ) -> None: ...
