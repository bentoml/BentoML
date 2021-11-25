import numpy as np
from pandas.core.arrays import (
    Categorical,
    DatetimeArray,
    IntervalArray,
    PeriodArray,
    TimedeltaArray,
)
from pandas.core.arrays._mixins import NDArrayBackedExtensionArray
from pandas.core.indexes.base import Index
from pandas.util._decorators import doc

"""
Shared methods for Index subclasses backed by ExtensionArray.
"""
_T = ...

def inherit_from_data(
    name: str, delegate, cache: bool = ..., wrap: bool = ...
):  # -> (self: Unknown, *args: Unknown, **kwargs: Unknown) -> Any | ABCDataFrame | Index | None:
    """
    Make an alias for a method of the underlying ExtensionArray.

    Parameters
    ----------
    name : str
        Name of an attribute the class should inherit from its EA parent.
    delegate : class
    cache : bool, default False
        Whether to convert wrapped properties into cache_readonly
    wrap : bool, default False
        Whether to wrap the inherited result in an Index.

    Returns
    -------
    attribute, method, property, or cache_readonly
    """
    ...

def inherit_names(
    names: list[str], delegate, cache: bool = ..., wrap: bool = ...
):  # -> (cls: Unknown) -> Unknown:
    """
    Class decorator to pin attributes from an ExtensionArray to a Index subclass.

    Parameters
    ----------
    names : List[str]
    delegate : class
    cache : bool, default False
    wrap : bool, default False
        Whether to wrap the inherited result in an Index.
    """
    ...

def make_wrapped_arith_op(opname: str): ...

class ExtensionIndex(Index):
    """
    Index subclass for indexes backed by ExtensionArray.
    """

    _data: IntervalArray | NDArrayBackedExtensionArray
    _data_cls: (
        type[Categorical]
        | type[DatetimeArray]
        | type[TimedeltaArray]
        | type[PeriodArray]
        | type[IntervalArray]
    )
    __eq__ = ...
    __ne__ = ...
    __lt__ = ...
    __gt__ = ...
    __le__ = ...
    __ge__ = ...
    def __getitem__(self, key): ...
    def searchsorted(self, value, side=..., sorter=...) -> np.ndarray: ...
    def putmask(self, mask, value) -> Index: ...
    def delete(self, loc):  # -> Self@ExtensionIndex:
        """
        Make new Index with passed location(-s) deleted

        Returns
        -------
        new_index : Index
        """
        ...
    def repeat(self, repeats, axis=...): ...
    def insert(self, loc: int, item) -> Index:
        """
        Make new Index inserting new item at location. Follows
        Python list.append semantics for negative values.

        Parameters
        ----------
        loc : int
        item : object

        Returns
        -------
        new_index : Index
        """
        ...
    @doc(Index.map)
    def map(self, mapper, na_action=...): ...
    @doc(Index.astype)
    def astype(self, dtype, copy: bool = ...) -> Index: ...
    @doc(Index.equals)
    def equals(self, other) -> bool: ...

class NDArrayBackedExtensionIndex(ExtensionIndex):
    """
    Index subclass for indexes backed by NDArrayBackedExtensionArray.
    """

    _data: NDArrayBackedExtensionArray
    ...
