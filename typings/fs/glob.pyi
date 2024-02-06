from typing import Any
from typing import Iterator
from typing import List
from typing import NamedTuple
from typing import Optional
from typing import Pattern
from typing import Text
from typing import Tuple

from .base import FS
from .info import Info
from .lrucache import LRUCache

class GlobMatch(NamedTuple):
    path: Text
    info: Info

class Counts(NamedTuple):
    files: int
    directories: int
    data: int

class LineCounts(NamedTuple):
    lines: int
    non_blank: int

_PATTERN_CACHE: LRUCache[Tuple[Text, bool], Tuple[int, bool, Pattern[Any]]] = ...

def match(pattern: str, path: str) -> bool: ...
def imatch(pattern: str, path: str) -> bool: ...

class Globber:
    def __init__(
        self,
        fs: FS,
        pattern: str,
        path: str = ...,
        namespaces: Optional[List[str]] = ...,
        case_sensitive: bool = ...,
        exclude_dirs: Optional[List[str]] = ...,
    ) -> None: ...
    def __repr__(self) -> Text: ...
    def __iter__(self) -> Iterator[GlobMatch]: ...
    def count(self) -> Counts: ...
    def count_lines(self) -> LineCounts: ...
    def remove(self) -> int: ...

class BoundGlobber:
    __slots__ = ["fs"]
    def __init__(self, fs: FS) -> None: ...
    def __repr__(self) -> Text: ...
    def __call__(
        self,
        pattern: str,
        path: str = ...,
        namespaces: Optional[List[str]] = ...,
        case_sensitive: bool = ...,
        exclude_dirs: Optional[List[str]] = ...,
    ) -> Globber: ...
