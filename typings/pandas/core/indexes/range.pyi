from typing import TYPE_CHECKING, Any, Hashable

import numpy as np
from pandas._typing import Dtype
from pandas.core.indexes.numeric import Int64Index, NumericIndex
from pandas.core.ops.common import unpack_zerodim_and_defer
from pandas.util._decorators import cache_readonly, doc

if TYPE_CHECKING: ...
_empty_range = ...

class RangeIndex(NumericIndex):
    """
    Immutable Index implementing a monotonic integer range.

    RangeIndex is a memory-saving special case of Int64Index limited to
    representing monotonic ranges. Using RangeIndex may in some instances
    improve computing speed.

    This is the default index type used
    by DataFrame and Series when no explicit index is provided by the user.

    Parameters
    ----------
    start : int (default: 0), range, or other RangeIndex instance
        If int and "stop" is not given, interpreted as "stop" instead.
    stop : int (default: 0)
    step : int (default: 1)
    dtype : np.int64
        Unused, accepted for homogeneity with other index types.
    copy : bool, default False
        Unused, accepted for homogeneity with other index types.
    name : object, optional
        Name to be stored in the index.

    Attributes
    ----------
    start
    stop
    step

    Methods
    -------
    from_range

    See Also
    --------
    Index : The base pandas Index type.
    Int64Index : Index of int64 data.
    """

    _typ = ...
    _engine_type = ...
    _dtype_validation_metadata = ...
    _range: range
    def __new__(
        cls,
        start=...,
        stop=...,
        step=...,
        dtype: Dtype | None = ...,
        copy: bool = ...,
        name: Hashable = ...,
    ) -> RangeIndex: ...
    @classmethod
    def from_range(cls, data: range, name=..., dtype: Dtype | None = ...) -> RangeIndex:
        """
        Create RangeIndex from a range object.

        Returns
        -------
        RangeIndex
        """
        ...
    def __reduce__(self): ...
    _deprecation_message = ...
    @property
    def start(self) -> int:
        """
        The value of the `start` parameter (``0`` if this was not supplied).
        """
        ...
    @property
    def stop(self) -> int:
        """
        The value of the `stop` parameter.
        """
        ...
    @property
    def step(self) -> int:
        """
        The value of the `step` parameter (``1`` if this was not supplied).
        """
        ...
    @cache_readonly
    def nbytes(self) -> int:
        """
        Return the number of bytes in the underlying data.
        """
        ...
    def memory_usage(self, deep: bool = ...) -> int:
        """
        Memory usage of my values

        Parameters
        ----------
        deep : bool
            Introspect the data deeply, interrogate
            `object` dtypes for system-level memory consumption

        Returns
        -------
        bytes used

        Notes
        -----
        Memory usage does not include memory consumed by elements that
        are not components of the array if deep=False

        See Also
        --------
        numpy.ndarray.nbytes
        """
        ...
    @property
    def dtype(self) -> np.dtype: ...
    @property
    def is_unique(self) -> bool:
        """return if the index has unique values"""
        ...
    @cache_readonly
    def is_monotonic_increasing(self) -> bool: ...
    @cache_readonly
    def is_monotonic_decreasing(self) -> bool: ...
    def __contains__(self, key: Any) -> bool: ...
    @property
    def inferred_type(self) -> str: ...
    @doc(Int64Index.get_loc)
    def get_loc(self, key, method=..., tolerance=...): ...
    def repeat(self, repeats, axis=...) -> Int64Index: ...
    def delete(self, loc) -> Int64Index: ...
    def take(
        self, indices, axis: int = ..., allow_fill: bool = ..., fill_value=..., **kwargs
    ) -> Int64Index: ...
    def tolist(self) -> list[int]: ...
    @doc(Int64Index.__iter__)
    def __iter__(self): ...
    @doc(Int64Index.copy)
    def copy(
        self, name: Hashable = ..., deep: bool = ..., dtype: Dtype | None = ..., names=...
    ): ...
    def min(self, axis=..., skipna: bool = ..., *args, **kwargs) -> int:
        """The minimum value of the RangeIndex"""
        ...
    def max(self, axis=..., skipna: bool = ..., *args, **kwargs) -> int:
        """The maximum value of the RangeIndex"""
        ...
    def argsort(self, *args, **kwargs) -> np.ndarray:
        """
        Returns the indices that would sort the index and its
        underlying data.

        Returns
        -------
        np.ndarray[np.intp]

        See Also
        --------
        numpy.ndarray.argsort
        """
        ...
    def factorize(
        self, sort: bool = ..., na_sentinel: int | None = ...
    ) -> tuple[np.ndarray, RangeIndex]: ...
    def equals(self, other: object) -> bool:
        """
        Determines if two Index objects contain the same elements.
        """
        ...
    def symmetric_difference(self, other, result_name: Hashable = ..., sort=...): ...
    def __len__(self) -> int:
        """
        return the length of the RangeIndex
        """
        ...
    @property
    def size(self) -> int: ...
    def __getitem__(self, key):  # -> RangeIndex | int | ExtensionArray | Any:
        """
        Conserve RangeIndex type for scalar and slice keys.
        """
        ...
    @unpack_zerodim_and_defer("__floordiv__")
    def __floordiv__(self, other): ...
    def all(self, *args, **kwargs) -> bool: ...
    def any(self, *args, **kwargs) -> bool: ...
