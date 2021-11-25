from typing import Any

from pandas.core.base import PandasObject

"""
frozen (immutable) data structures to support MultiIndexing

These are used for:

- .names (FrozenList)

"""

class FrozenList(PandasObject, list):
    """
    Container that doesn't allow setting item *but*
    because it's technically non-hashable, will be used
    for lookups, appropriately, etc.
    """

    def union(self, other) -> FrozenList:
        """
        Returns a FrozenList with other concatenated to the end of self.

        Parameters
        ----------
        other : array-like
            The array-like whose elements we are concatenating.

        Returns
        -------
        FrozenList
            The collection difference between self and other.
        """
        ...
    def difference(self, other) -> FrozenList:
        """
        Returns a FrozenList with elements from other removed from self.

        Parameters
        ----------
        other : array-like
            The array-like whose elements we are removing self.

        Returns
        -------
        FrozenList
            The collection difference between self and other.
        """
        ...
    __add__ = ...
    def __getitem__(self, n): ...
    def __radd__(self, other): ...
    def __eq__(self, other: Any) -> bool: ...
    __req__ = ...
    def __mul__(self, other): ...
    __imul__ = ...
    def __reduce__(self): ...
    def __hash__(self) -> int: ...
    def __str__(self) -> str: ...
    def __repr__(self) -> str: ...
    __setitem__ = ...
    __delitem__ = ...
    pop = ...
    remove = ...
