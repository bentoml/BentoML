import typing
from typing import Any
from typing import Callable
from typing import Collection
from typing import Iterator
from typing import List
from typing import NamedTuple
from typing import Optional
from typing import Text
from typing import Tuple
from typing import Type

from .base import FS
from .info import Info

if typing.TYPE_CHECKING:
    OnError = Callable[[Text, Exception], bool]
_F = typing.TypeVar("_F", bound="FS")

class Step(NamedTuple):
    path: Text
    dirs: List[Info]
    files: List[Info]

class Walker:
    def __init__(
        self,
        ignore_errors: bool = ...,
        on_error: Optional[OnError] = ...,
        search: Text = ...,
        filter: Optional[List[Text]] = ...,
        exclude: Optional[List[Text]] = ...,
        filter_dirs: Optional[List[Text]] = ...,
        exclude_dirs: Optional[List[Text]] = ...,
        max_depth: Optional[int] = ...,
    ) -> None: ...
    @classmethod
    def bind(cls, fs: _F) -> BoundWalker[_F]: ...
    def __repr__(self) -> Text: ...
    def check_open_dir(self, fs: FS, path: Text, info: Info) -> bool: ...
    def check_scan_dir(self, fs: FS, path: Text, info: Info) -> bool: ...
    def check_file(self, fs: FS, info: Info) -> bool: ...
    def walk(
        self, fs: FS, path: Text = ..., namespaces: Optional[Collection[Text]] = ...
    ) -> Iterator[Step]: ...
    def files(self, fs: FS, path: Text = ...) -> Iterator[Text]: ...
    def dirs(self, fs: FS, path: Text = ...) -> Iterator[Text]: ...
    def info(
        self, fs: FS, path: Text = ..., namespaces: Optional[Collection[Text]] = ...
    ) -> Iterator[Tuple[Text, Info]]: ...

class BoundWalker(typing.Generic[_F]):
    def __init__(self, fs: _F, walker_class: Type[Walker] = ...) -> None: ...
    def __repr__(self) -> Text: ...
    def walk(
        self,
        path: Text = ...,
        namespaces: Optional[Collection[Text]] = ...,
        **kwargs: Any,
    ) -> Iterator[Step]: ...

    __call__ = walk

    def files(self, path: Text = ..., **kwargs: Any) -> Iterator[Text]: ...
    def dirs(self, path: Text = ..., **kwargs: Any) -> Iterator[Text]: ...
    def info(
        self,
        path: Text = ...,
        namespaces: Optional[Collection[Text]] = ...,
        **kwargs: Any,
    ) -> Iterator[Tuple[Text, Info]]: ...

default_walker: Walker = ...
walk = default_walker.walk
walk_files = default_walker.files
walk_info = default_walker.info
walk_dirs = default_walker.dirs
