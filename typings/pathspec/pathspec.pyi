from collections.abc import Callable
from collections.abc import Collection
from collections.abc import Iterable
from collections.abc import Iterator
from os import PathLike
from typing import Any
from typing import AnyStr
from typing import NoReturn

from .pattern import Pattern
from .util import TreeEntry

class PathSpec:
    def __init__(self, patterns: Iterable[Pattern]) -> None: ...
    def __eq__(self, other: PathSpec) -> bool: ...
    def __len__(self) -> int: ...
    def __add__(self, other: PathSpec) -> PathSpec: ...
    def __iadd__(self, other: PathSpec) -> PathSpec: ...
    @classmethod
    def from_lines(
        cls,
        pattern_factory: str | Callable[[AnyStr], Pattern],
        lines: Iterable[AnyStr],
    ) -> PathSpec: ...
    def match_file(
        self,
        file: str | PathLike[Any],
        separators: Collection[str] | None = ...,
    ) -> bool: ...
    def match_entries(
        self, entries: Iterable[TreeEntry], separators: Collection[str] | None = ...
    ) -> Iterator[TreeEntry]: ...
    def match_files(
        self,
        files: Iterable[str | PathLike[str]],
        separators: Collection[str] | None = ...,
    ) -> Iterator[str | PathLike[str]]: ...
    def match_tree_entries(
        self,
        root: str,
        on_error: Callable[[type[Exception]], NoReturn] | None = ...,
        follow_links: bool | None = ...,
    ) -> Iterator[TreeEntry]: ...
    def match_tree_files(
        self,
        root: str,
        on_error: Callable[[type[Exception]], NoReturn] | None = ...,
        follow_links: bool | None = ...,
    ) -> Iterator[str]: ...

    match_tree: Callable[
        [str, Callable[[type[Exception]], NoReturn] | None, bool | None],
        Iterator[str],
    ] = ...
