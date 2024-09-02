from __future__ import annotations

from os import PathLike
from typing import Any
from typing import AnyStr
from typing import Callable
from typing import Collection
from typing import Iterable
from typing import Iterator
from typing import NoReturn
from typing import Text

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
        pattern_factory: Text | Callable[[AnyStr], Pattern],
        lines: Iterable[AnyStr],
    ) -> PathSpec: ...
    def match_file(
        self,
        file: Text | PathLike[Any],
        separators: Collection[Text] | None = ...,
    ) -> bool: ...
    def match_entries(
        self, entries: Iterable[TreeEntry], separators: Collection[Text] | None = ...
    ) -> Iterator[TreeEntry]: ...
    def match_files(
        self,
        files: Iterable[Text | PathLike[str]],
        separators: Collection[Text] | None = ...,
    ) -> Iterator[Text | PathLike[str]]: ...
    def match_tree_entries(
        self,
        root: Text,
        on_error: Callable[[type[Exception]], NoReturn] | None = ...,
        follow_links: bool | None = ...,
    ) -> Iterator[TreeEntry]: ...
    def match_tree_files(
        self,
        root: Text,
        on_error: Callable[[type[Exception]], NoReturn] | None = ...,
        follow_links: bool | None = ...,
    ) -> Iterator[Text]: ...

    match_tree: Callable[
        [Text, Callable[[type[Exception]], NoReturn] | None, bool | None],
        Iterator[Text],
    ] = ...
