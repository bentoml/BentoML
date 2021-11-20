"""
This type stub file was generated by pyright.
"""

from pathlib import Path
from typing import Any, Callable, Optional, Type, Union

from .parse import Protocol
from .types import StrBytes

__all__ = ("parse_file_as", "parse_obj_as", "parse_raw_as")
NameFactory = Union[str, Callable[[Type[Any]], str]]
T = ...

def parse_obj_as(
    type_: Type[T], obj: Any, *, type_name: Optional[NameFactory] = ...
) -> T: ...
def parse_file_as(
    type_: Type[T],
    path: Union[str, Path],
    *,
    content_type: str = ...,
    encoding: str = ...,
    proto: Protocol = ...,
    allow_pickle: bool = ...,
    json_loads: Callable[[str], Any] = ...,
    type_name: Optional[NameFactory] = ...
) -> T: ...
def parse_raw_as(
    type_: Type[T],
    b: StrBytes,
    *,
    content_type: str = ...,
    encoding: str = ...,
    proto: Protocol = ...,
    allow_pickle: bool = ...,
    json_loads: Callable[[str], Any] = ...,
    type_name: Optional[NameFactory] = ...
) -> T: ...
