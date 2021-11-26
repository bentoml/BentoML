from typing import Pattern, Sequence
from pandas._typing import FilePathOrBuffer
from pandas.core.frame import DataFrame
from pandas.util._decorators import deprecate_nonkeyword_arguments

_IMPORTS = ...
_HAS_BS4 = ...
_HAS_LXML = ...
_HAS_HTML5LIB = ...
_RE_WHITESPACE = ...

class _HtmlFrameParser:
    def __init__(self, io, match, attrs, encoding, displayed_only) -> None: ...
    def parse_tables(self): ...

class _BeautifulSoupHtml5LibFrameParser(_HtmlFrameParser):
    def __init__(self, *args, **kwargs) -> None: ...

_re_namespace = ...
_valid_schemes = ...

class _LxmlFrameParser(_HtmlFrameParser):
    def __init__(self, *args, **kwargs) -> None: ...

_valid_parsers = ...

@deprecate_nonkeyword_arguments(version="2.0")
def read_html(
    io: FilePathOrBuffer,
    match: str | Pattern = ...,
    flavor: str | None = ...,
    header: int | Sequence[int] | None = ...,
    index_col: int | Sequence[int] | None = ...,
    skiprows: int | Sequence[int] | slice | None = ...,
    attrs: dict[str, str] | None = ...,
    parse_dates: bool = ...,
    thousands: str | None = ...,
    encoding: str | None = ...,
    decimal: str = ...,
    converters: dict | None = ...,
    na_values=...,
    keep_default_na: bool = ...,
    displayed_only: bool = ...,
) -> list[DataFrame]: ...
