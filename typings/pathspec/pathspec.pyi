from os import PathLike
from typing import (
    Any,
    AnyStr,
    Callable,
    Collection,
    Iterable,
    Iterator,
    Optional,
    Text,
    Union,
)
from .pattern import Pattern
from .util import TreeEntry

class PathSpec:
    def __init__(self, patterns: Iterable[Pattern]) -> None: ...
    def __eq__(self, other: PathSpec) -> bool: ...
    def __len__(self): ...
    def __add__(self, other: PathSpec) -> PathSpec: ...
    def __iadd__(self, other: PathSpec) -> PathSpec: ...
    @classmethod
    def from_lines(
        cls,
        pattern_factory: Union[Text, Callable[[AnyStr], Pattern]],
        lines: Iterable[AnyStr],
    ) -> PathSpec: ...
    def match_file(
        self,
        file: Union[Text, PathLike[Any]],
        separators: Optional[Collection[Text]] = ...,
    ) -> bool: ...
    def match_entries(
        self, entries: Iterable[TreeEntry], separators: Optional[Collection[Text]] = ...
    ) -> Iterator[TreeEntry]: ...
    def match_files(
        self,
        files: Iterable[Union[Text, PathLike]],
        separators: Optional[Collection[Text]] = ...,
    ) -> Iterator[Union[Text, PathLike]]: ...
    def match_tree_entries(
        self,
        root: Text,
        on_error: Optional[Callable] = ...,
        follow_links: Optional[bool] = ...,
    ) -> Iterator[TreeEntry]: ...
    def match_tree_files(
        self,
        root: Text,
        on_error: Optional[Callable] = ...,
        follow_links: Optional[bool] = ...,
    ) -> Iterator[Text]: ...
    match_tree = ...
