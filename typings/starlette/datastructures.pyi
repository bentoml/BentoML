"""
This type stub file was generated by pyright.
"""

import typing as t
from collections.abc import Sequence
from urllib.parse import SplitResult

from starlette.types import Scope

Address = ...

class URL:
    def __init__(self, url: str = ..., scope: Scope = ..., **components: t.Any) -> None:
        ...

    @property
    def components(self) -> SplitResult:
        ...

    @property
    def scheme(self) -> str:
        ...

    @property
    def netloc(self) -> str:
        ...

    @property
    def path(self) -> str:
        ...

    @property
    def query(self) -> str:
        ...

    @property
    def fragment(self) -> str:
        ...

    @property
    def username(self) -> t.Union[None, str]:
        ...

    @property
    def password(self) -> t.Union[None, str]:
        ...

    @property
    def hostname(self) -> t.Union[None, str]:
        ...

    @property
    def port(self) -> t.Optional[int]:
        ...

    @property
    def is_secure(self) -> bool:
        ...

    def replace(self, **kwargs: t.Any) -> URL:
        ...

    def include_query_params(self, **kwargs: t.Any) -> URL:
        ...

    def replace_query_params(self, **kwargs: t.Any) -> URL:
        ...

    def remove_query_params(self, keys: t.Union[str, t.Sequence[str]]) -> URL:
        ...

    def __eq__(self, other: t.Any) -> bool:
        ...

    def __str__(self) -> str:
        ...

    def __repr__(self) -> str:
        ...



class URLPath(str):
    """
    A URL path string that may also hold an associated protocol and/or host.
    Used by the routing to return `url_path_for` matches.
    """
    def __new__(cls, path: str, protocol: str = ..., host: str = ...) -> URLPath:
        ...

    def __init__(self, path: str, protocol: str = ..., host: str = ...) -> None:
        ...

    def make_absolute_url(self, base_url: t.Union[str, URL]) -> str:
        ...



class Secret:
    """
    Holds a string value that should not be revealed in tracebacks etc.
    You should cast the value to `str` at the point it is required.
    """
    def __init__(self, value: str) -> None:
        ...

    def __repr__(self) -> str:
        ...

    def __str__(self) -> str:
        ...



class CommaSeparatedStrings(Sequence):
    def __init__(self, value: t.Union[str, t.Sequence[str]]) -> None:
        ...

    def __len__(self) -> int:
        ...

    def __getitem__(self, index: t.Union[int, slice]) -> t.Any:
        ...

    def __iter__(self) -> t.Iterator[str]:
        ...

    def __repr__(self) -> str:
        ...

    def __str__(self) -> str:
        ...



class ImmutableMultiDict(t.Mapping):
    def __init__(self, *args: t.Union[ImmutableMultiDict, t.Mapping, t.List[t.Tuple[t.Any, t.Any]]],, **kwargs: t.Any) -> None:
        ...

    def getlist(self, key: t.Any) -> t.List[str]:
        ...

    def keys(self) -> t.KeysView[t.Any]:
        ...

    def values(self) -> t.ValuesView[t.Any]:
        ...

    def items(self) -> t.ItemsView[t.Any, t.Any]:
        ...

    def multi_items(self) -> t.List[t.Tuple[str, str]]:
        ...

    def get(self, key: t.Any, default: t.Any = ...) -> t.Any:
        ...

    def __getitem__(self, key: t.Any) -> str:
        ...

    def __contains__(self, key: t.Any) -> bool:
        ...

    def __iter__(self) -> t.Iterator[t.Any]:
        ...

    def __len__(self) -> int:
        ...

    def __eq__(self, other: t.Any) -> bool:
        ...

    def __repr__(self) -> str:
        ...



class MultiDict(ImmutableMultiDict):
    def __setitem__(self, key: t.Any, value: t.Any) -> None:
        ...

    def __delitem__(self, key: t.Any) -> None:
        ...

    def pop(self, key: t.Any, default: t.Any = ...) -> t.Any:
        ...

    def popitem(self) -> t.Tuple:
        ...

    def poplist(self, key: t.Any) -> t.List:
        ...

    def clear(self) -> None:
        ...

    def setdefault(self, key: t.Any, default: t.Any = ...) -> t.Any:
        ...

    def setlist(self, key: t.Any, values: t.List) -> None:
        ...

    def append(self, key: t.Any, value: t.Any) -> None:
        ...

    def update(self, *args: t.Union[MultiDict, t.Mapping, t.List[t.Tuple[t.Any, t.Any]]],, **kwargs: t.Any) -> None:
        ...



class QueryParams(ImmutableMultiDict):
    """
    An immutable multidict.
    """
    def __init__(self, *args: t.Union[ImmutableMultiDict, t.Mapping, t.List[t.Tuple[t.Any, t.Any]], str, bytes],, **kwargs: t.Any) -> None:
        ...

    def __str__(self) -> str:
        ...

    def __repr__(self) -> str:
        ...



class UploadFile:
    """
    An uploaded file included as part of the request data.
    """
    spool_max_size = ...
    def __init__(self, filename: str, file: t.IO = ..., content_type: str = ...) -> None:
        ...

    async def write(self, data: t.Union[bytes, str]) -> None:
        ...

    async def read(self, size: int = ...) -> t.Union[bytes, str]:
        ...

    async def seek(self, offset: int) -> None:
        ...

    async def close(self) -> None:
        ...



class FormData(ImmutableMultiDict):
    """
    An immutable multidict, containing both file uploads and text input.
    """
    def __init__(self, *args: t.Union[FormData, t.Mapping[str, t.Union[str, UploadFile]], t.List[t.Tuple[str, t.Union[str, UploadFile]]]], **kwargs: t.Union[str, UploadFile]) -> None:
        ...

    async def close(self) -> None:
        ...



class Headers(t.Mapping[str, str]):
    """
    An immutable, case-insensitive multidict.
    """
    def __init__(self, headers: t.Mapping[str, str] = ..., raw: t.List[t.Tuple[bytes, bytes]] = ..., scope: Scope = ...) -> None:
        ...

    @property
    def raw(self) -> t.List[t.Tuple[bytes, bytes]]:
        ...

    def keys(self) -> t.List[str]:
        ...

    def values(self) -> t.List[str]:
        ...

    def items(self) -> t.List[t.Tuple[str, str]]:
        ...

    def get(self, key: str, default: t.Any = ...) -> t.Any:
        ...

    def getlist(self, key: str) -> t.List[str]:
        ...

    def mutablecopy(self) -> MutableHeaders:
        ...

    def __getitem__(self, key: str) -> str:
        ...

    def __contains__(self, key: t.Any) -> bool:
        ...

    def __iter__(self) -> t.Iterator[t.Any]:
        ...

    def __len__(self) -> int:
        ...

    def __eq__(self, other: t.Any) -> bool:
        ...

    def __repr__(self) -> str:
        ...



class MutableHeaders(Headers):
    def __setitem__(self, key: str, value: str) -> None:
        """
        Set the header `key` to `value`, removing any duplicate entries.
        Retains insertion order.
        """
        ...

    def __delitem__(self, key: str) -> None:
        """
        Remove the header `key`.
        """
        ...

    @property
    def raw(self) -> t.List[t.Tuple[bytes, bytes]]:
        ...

    def setdefault(self, key: str, value: str) -> str:
        """
        If the header `key` does not exist, then set it to `value`.
        Returns the header value.
        """
        ...

    def update(self, other: dict) -> None:
        ...

    def append(self, key: str, value: str) -> None:
        """
        Append a header, preserving any duplicate entries.
        """
        ...

    def add_vary_header(self, vary: str) -> None:
        ...



class State:
    """
    An object that can be used to store arbitrary state.

    Used for `request.state` and `app.state`.
    """
    def __init__(self, state: t.Dict = ...) -> None:
        ...

    def __setattr__(self, key: t.Any, value: t.Any) -> None:
        ...

    def __getattr__(self, key: t.Any) -> t.Any:
        ...

    def __delattr__(self, key: t.Any) -> None:
        ...



