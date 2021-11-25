

import typing
from collections import OrderedDict

"""Least Recently Used cache mapping.
"""
_K = typing.TypeVar("_K")
_V = typing.TypeVar("_V")

class LRUCache(OrderedDict[_K, _V], typing.Generic[_K, _V]):
    """A dictionary-like container that stores a given maximum items.

    If an additional item is added when the LRUCache is full, the least
    recently used key is discarded to make room for the new item.

    """

    def __init__(self, cache_size: int) -> None:
        """Create a new LRUCache with the given size."""
        ...
    def __setitem__(self, key: _K, value: _V) -> None:
        """Store a new views, potentially discarding an old value."""
        ...
    def __getitem__(self, key: _K) -> _V:
        """Get the item, but also makes it most recent."""
        ...
