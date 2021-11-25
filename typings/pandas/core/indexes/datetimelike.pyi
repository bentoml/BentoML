from typing import TYPE_CHECKING, Any

import numpy as np
from pandas._libs.tslibs import BaseOffset, NaTType, Resolution
from pandas._typing import Callable
from pandas.core.arrays import DatetimeArray, PeriodArray, TimedeltaArray
from pandas.core.arrays.datetimelike import DatetimeLikeArrayMixin
from pandas.core.indexes.base import Index, _index_shared_docs
from pandas.core.indexes.extension import NDArrayBackedExtensionIndex, inherit_names
from pandas.util._decorators import Appender, doc

"""
Base and utility classes for tseries type pandas objects.
"""
if TYPE_CHECKING: ...
_index_doc_kwargs = ...
_T = ...

@inherit_names(
    ["inferred_freq", "_resolution_obj", "resolution"], DatetimeLikeArrayMixin, cache=True
)
@inherit_names(["mean", "asi8", "freq", "freqstr"], DatetimeLikeArrayMixin)
class DatetimeIndexOpsMixin(NDArrayBackedExtensionIndex):
    """
    Common ops mixin to support a unified interface datetimelike Index.
    """

    _is_numeric_dtype = ...
    _can_hold_strings = ...
    _data: DatetimeArray | TimedeltaArray | PeriodArray
    freq: BaseOffset | None
    freqstr: str | None
    _resolution_obj: Resolution
    _bool_ops: list[str] = ...
    _field_ops: list[str] = ...
    hasnans = ...
    _hasnans = ...
    @property
    def values(self) -> np.ndarray: ...
    def __array_wrap__(
        self, result, context=...
    ):  # -> NDArrayBackedExtensionIndex | object | Index | DatetimeTimedeltaMixin:
        """
        Gets called after a ufunc and other functions.
        """
        ...
    def equals(self, other: Any) -> bool:
        """
        Determines if two Index objects contain the same elements.
        """
        ...
    @Appender(Index.__contains__.__doc__)
    def __contains__(self, key: Any) -> bool: ...
    @Appender(_index_shared_docs["take"] % _index_doc_kwargs)
    def take(self, indices, axis=..., allow_fill=..., fill_value=..., **kwargs): ...
    _can_hold_na = ...
    _na_value: NaTType = ...
    def tolist(self) -> list:
        """
        Return a list of the underlying data.
        """
        ...
    def min(
        self, axis=..., skipna=..., *args, **kwargs
    ):  # -> NaTType | Timestamp | Timedelta | Period:
        """
        Return the minimum value of the Index or minimum along
        an axis.

        See Also
        --------
        numpy.ndarray.min
        Series.min : Return the minimum value in a Series.
        """
        ...
    def argmin(self, axis=..., skipna=..., *args, **kwargs):  # -> Literal[-1]:
        """
        Returns the indices of the minimum values along an axis.

        See `numpy.ndarray.argmin` for more information on the
        `axis` parameter.

        See Also
        --------
        numpy.ndarray.argmin
        """
        ...
    def max(
        self, axis=..., skipna=..., *args, **kwargs
    ):  # -> NaTType | Timestamp | Timedelta | Period:
        """
        Return the maximum value of the Index or maximum along
        an axis.

        See Also
        --------
        numpy.ndarray.max
        Series.max : Return the maximum value in a Series.
        """
        ...
    def argmax(self, axis=..., skipna=..., *args, **kwargs):  # -> Literal[-1]:
        """
        Returns the indices of the maximum values along an axis.

        See `numpy.ndarray.argmax` for more information on the
        `axis` parameter.

        See Also
        --------
        numpy.ndarray.argmax
        """
        ...
    def format(
        self,
        name: bool = ...,
        formatter: Callable | None = ...,
        na_rep: str = ...,
        date_format: str | None = ...,
    ) -> list[str]:
        """
        Render a string representation of the Index.
        """
        ...
    __add__ = ...
    __sub__ = ...
    __radd__ = ...
    __rsub__ = ...
    __pow__ = ...
    __rpow__ = ...
    __mul__ = ...
    __rmul__ = ...
    __floordiv__ = ...
    __rfloordiv__ = ...
    __mod__ = ...
    __rmod__ = ...
    __divmod__ = ...
    __rdivmod__ = ...
    __truediv__ = ...
    __rtruediv__ = ...
    def shift(self: _T, periods: int = ..., freq=...) -> _T:
        """
        Shift index by desired number of time frequency increments.

        This method is for shifting the values of datetime-like indexes
        by a specified time increment a given number of times.

        Parameters
        ----------
        periods : int, default 1
            Number of periods (or increments) to shift by,
            can be positive or negative.
        freq : pandas.DateOffset, pandas.Timedelta or string, optional
            Frequency increment to shift by.
            If None, the index is shifted by its own `freq` attribute.
            Offset aliases are valid strings, e.g., 'D', 'W', 'M' etc.

        Returns
        -------
        pandas.DatetimeIndex
            Shifted index.

        See Also
        --------
        Index.shift : Shift values of Index.
        PeriodIndex.shift : Shift values of PeriodIndex.
        """
        ...
    @doc(NDArrayBackedExtensionIndex.delete)
    def delete(self: _T, loc) -> _T: ...
    @doc(NDArrayBackedExtensionIndex.insert)
    def insert(self, loc: int, item): ...

class DatetimeTimedeltaMixin(DatetimeIndexOpsMixin):
    """
    Mixin class for methods shared by DatetimeIndex and TimedeltaIndex,
    but not PeriodIndex
    """

    _data: DatetimeArray | TimedeltaArray
    _comparables = ...
    _attributes = ...
    _is_monotonic_increasing = ...
    _is_monotonic_decreasing = ...
    _is_unique = ...
    def is_type_compatible(self, kind: str) -> bool: ...
    _join_precedence = ...
    def join(
        self,
        other,
        how: str = ...,
        level=...,
        return_indexers: bool = ...,
        sort: bool = ...,
    ):  # -> tuple[MultiIndex, Unknown, Unknown] | tuple[MultiIndex, ndarray | None, ndarray | None] | tuple[Self@Index, None, ndarray] | tuple[Index, ndarray, None] | tuple[Unknown, Unknown, Unknown] | tuple[Index, ndarray, ndarray] | tuple[Index, None, None] | tuple[Index | Unbound, ndarray | None, ndarray | None] | tuple[tuple[MultiIndex, Unknown | ndarray | Any] | MultiIndex | Unknown | Index | Unbound | None, ndarray | None, ndarray | None] | tuple[Self@DatetimeTimedeltaMixin, None, ndarray] | tuple[tuple[MultiIndex, Unknown | ndarray | Any] | MultiIndex | Unknown | Self@DatetimeTimedeltaMixin | <subclass of DatetimeTimedeltaMixin and MultiIndex>* | Index | Unbound | None, ndarray | None, ndarray | None]:
        """
        See Index.join
        """
        ...
