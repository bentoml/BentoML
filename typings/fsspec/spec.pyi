from typing import Any
from typing import ClassVar
from typing import Generator
from typing import Literal
from typing import TypedDict
from typing import overload

from fsspec.callbacks import Callback as Callback

class InfoDict(TypedDict):
    name: str
    size: int
    type: Literal["file", "directory"]

class AbstractFileSystem:
    cachable = True
    blocksize = 2**22
    sep = "/"
    protocol: ClassVar[str | tuple[str, ...]] = "abstract"
    async_impl = False

    def mkdir(self, path: str, create_parents: bool = True, **kwargs: Any) -> None: ...
    def makedirs(self, path: str, exist_ok: bool = False) -> None: ...
    def rmdir(self, path: str) -> None: ...
    @overload
    def ls(
        self, path: str, detail: Literal[True] = ..., **kwargs: Any
    ) -> list[InfoDict]: ...
    @overload
    def ls(
        self, path: str, detail: Literal[False] = ..., **kwargs: Any
    ) -> list[str]: ...
    def walk(
        self,
        path: str,
        maxdepth: int | None = None,
        topdown: bool = True,
        on_error: str = "omit",
        **kwargs: Any,
    ) -> Generator[tuple[str, list[str], list[str]], None, None]: ...
    @overload
    def find(
        self,
        path: str,
        maxdepth: int | None = None,
        withdirs: bool = False,
        detail: Literal[False] = ...,
        **kwargs: Any,
    ) -> list[str]: ...
    @overload
    def find(
        self,
        path: str,
        maxdepth: int | None = None,
        withdirs: bool = False,
        detail: Literal[True] = ...,
        **kwargs: Any,
    ) -> list[InfoDict]: ...
    def du(
        self,
        path: str,
        total: bool = True,
        maxdepth: int | None = None,
        withdirs: bool = False,
        **kwargs: Any,
    ) -> int: ...
    @overload
    def glob(
        self,
        path: str,
        maxdepth: int | None = None,
        detail: Literal[False] = ...,
        **kwargs: Any,
    ) -> list[str]: ...
    @overload
    def glob(
        self,
        path: str,
        maxdepth: int | None = None,
        detail: Literal[True] = ...,
        **kwargs: Any,
    ) -> list[InfoDict]: ...
    def exists(self, path: str, **kwargs: Any) -> bool: ...
    def lexists(self, path: str, **kwargs: Any) -> bool: ...
    def info(self, path: str, **kwargs: Any) -> InfoDict: ...
    def checksum(self, path: str) -> str: ...
    def size(self, path: str) -> int: ...
    def sizes(self, paths: list[str]) -> list[int]: ...
    def isdir(self, path: str) -> bool: ...
    def isfile(self, path: str) -> bool: ...
    def read_text(
        self,
        path: str,
        encoding: str | None = None,
        errors: str | None = None,
        newline: str | None = None,
        **kwargs: Any,
    ) -> str: ...
    def write_text(
        self,
        path: str,
        value: str,
        encoding: str | None = None,
        errors: str | None = None,
        newline: str | None = None,
        **kwargs: Any,
    ) -> int: ...
    def get_file(
        self,
        rpath: str,
        lpath: str,
        callback: Callback = ...,
        outfile: str | None = None,
        **kwargs: Any,
    ) -> None: ...
    def get(
        self,
        rpath: str | list[str],
        lpath: str | list[str],
        recursive: bool = False,
        callback: Callback = ...,
        maxdepth: int | None = None,
        **kwargs: Any,
    ) -> None: ...
    def put_file(
        self,
        lpath: str,
        rpath: str,
        callback: Callback = ...,
        mode: Literal["create", "overwrite"] = ...,
        **kwargs: Any,
    ) -> None: ...
    def put(
        self,
        lpath: str | list[str],
        rpath: str | list[str],
        recursive: bool = False,
        callback: Callback = ...,
        maxdepth: int | None = None,
        **kwargs: Any,
    ) -> None: ...
    def head(self, path: str, size: int = ...) -> None: ...
    def tail(self, path: str, size: int = ...) -> None: ...
    def copy(
        self,
        path1: str | list[str],
        path2: str | list[str],
        recursive: bool = False,
        maxdepth: int | None = None,
        on_error: str | None = None,
        **kwargs: Any,
    ) -> None: ...
    def expand_path(
        self,
        path: str | list[str],
        recursive: bool = False,
        maxdepth: int | None = None,
        **kwargs: Any,
    ) -> None: ...
    def rm_file(self, path: str) -> None: ...
    def rm(
        self,
        path: str | list[str],
        recursive: bool = False,
        maxdepth: int | None = None,
    ) -> None: ...
    def open(
        self,
        path: str,
        mode: str = "rb",
        block_size: int | None = None,
        cache_options: dict[str, Any] | None = None,
        compression: str | None = None,
        **kwargs: Any,
    ) -> None: ...
    def to_json(self, *, include_password: bool = True) -> str: ...
    @staticmethod
    def from_json(blob: str) -> AbstractFileSystem: ...
    def to_dict(self, *, include_password: bool = True) -> dict[str, Any]: ...
    @staticmethod
    def from_dict(dct: dict[str, Any]) -> AbstractFileSystem: ...
