import typing
from collections import OrderedDict

_K = typing.TypeVar("_K")
_V = typing.TypeVar("_V")

class LRUCache(OrderedDict[_K, _V], typing.Generic[_K, _V]):
    def __init__(self, cache_size: int) -> None: ...
    def __setitem__(self, key: _K, value: _V) -> None: ...
    def __getitem__(self, key: _K) -> _V: ...
