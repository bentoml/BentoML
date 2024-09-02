from __future__ import annotations

from io import BytesIO
from typing import IO
from typing import Any
from typing import Callable
from typing import Iterator
from typing import Text
from typing import TypeVar
from typing import Union
from typing import overload

from .dumper import Dumper as Dumper
from .error import YAMLError as YAMLError
from .nodes import Node
from .representer import BaseRepresenter as BaseRepresenter

_Yaml = Any
__with_libyaml__: Any
__version__: str
_T = TypeVar("_T")
_R = TypeVar("_R")

def safe_load(stream: Union[bytes, IO[bytes], Text, IO[Text]]) -> Any: ...
def safe_load_all(stream: Union[bytes, IO[bytes], Text, IO[Text]]) -> Iterator[Any]: ...
@overload
def dump(
    data: Any,
    stream: IO[str],
    Dumper: type[Dumper] = ...,
    *,
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
@overload
def dump(
    data: Any,
    stream: None = ...,
    Dumper: type[Dumper] = ...,
    *,
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
) -> _Yaml: ...
@overload
def safe_dump(
    data: Any,
    stream: BytesIO | None,
    *,
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
@overload
def safe_dump(
    data: Any,
    stream: None = ...,
    *,
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
) -> BytesIO: ...
def add_representer(
    data_type: type[_T],
    representer: Callable[[_R, _T], Node],
    Dumper: type[_R] = ...,
) -> None: ...
