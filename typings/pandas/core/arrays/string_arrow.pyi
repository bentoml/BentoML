from typing import TYPE_CHECKING, Any, Sequence

import numpy as np
from pandas import Series
from pandas._typing import NpDtype, PositionalIndexer
from pandas.compat import pa_version_under1p0
from pandas.core.arraylike import OpsMixin
from pandas.core.arrays.base import ExtensionArray
from pandas.core.arrays.string_ import BaseStringArray, StringDtype
from pandas.core.strings.object_array import ObjectStringArrayMixin
from pandas.util._decorators import doc

if notpa_version_under1p0:
    ARROW_CMP_FUNCS = ...
if TYPE_CHECKING: ...

class ArrowStringArray(OpsMixin, BaseStringArray, ObjectStringArrayMixin):
    """
    Extension array for string data in a ``pyarrow.ChunkedArray``.

    .. versionadded:: 1.2.0

    .. warning::

       ArrowStringArray is considered experimental. The implementation and
       parts of the API may change without warning.

    Parameters
    ----------
    values : pyarrow.Array or pyarrow.ChunkedArray
        The array of data.

    Attributes
    ----------
    None

    Methods
    -------
    None

    See Also
    --------
    array
        The recommended function for creating a ArrowStringArray.
    Series.str
        The string methods are available on Series backed by
        a ArrowStringArray.

    Notes
    -----
    ArrowStringArray returns a BooleanArray for comparison methods.

    Examples
    --------
    >>> pd.array(['This is', 'some text', None, 'data.'], dtype="string[pyarrow]")
    <ArrowStringArray>
    ['This is', 'some text', <NA>, 'data.']
    Length: 4, dtype: string
    """

    def __init__(self, values) -> None: ...
    @property
    def dtype(self) -> StringDtype:
        """
        An instance of 'string[pyarrow]'.
        """
        ...
    def __array__(self, dtype: NpDtype | None = ...) -> np.ndarray:
        """Correctly construct numpy arrays when passed to `np.asarray()`."""
        ...
    def __arrow_array__(self, type=...):
        """Convert myself to a pyarrow Array or ChunkedArray."""
        ...
    def to_numpy(
        self, dtype: NpDtype | None = ..., copy: bool = ..., na_value=...
    ) -> np.ndarray:
        """
        Convert to a NumPy ndarray.
        """
        ...
    def __len__(self) -> int:
        """
        Length of this array.

        Returns
        -------
        length : int
        """
        ...
    @doc(ExtensionArray.factorize)
    def factorize(self, na_sentinel: int = ...) -> tuple[np.ndarray, ExtensionArray]: ...
    def __getitem__(self, item: PositionalIndexer) -> Any:
        """Select a subset of self.

        Parameters
        ----------
        item : int, slice, or ndarray
            * int: The position in 'self' to get.
            * slice: A slice object, where 'start', 'stop', and 'step' are
              integers or None
            * ndarray: A 1-d boolean NumPy ndarray the same length as 'self'

        Returns
        -------
        item : scalar or ExtensionArray

        Notes
        -----
        For scalar ``item``, return a scalar value suitable for the array's
        type. This should be an instance of ``self.dtype.type``.
        For slice ``key``, return an instance of ``ExtensionArray``, even
        if the slice is length 0 or 1.
        For a boolean mask, return an instance of ``ExtensionArray``, filtered
        to the values where ``item`` is True.
        """
        ...
    def fillna(
        self, value=..., method=..., limit=...
    ):  # -> Self@ArrowStringArray | ArrowStringArray:
        """
        Fill NA/NaN values using the specified method.

        Parameters
        ----------
        value : scalar, array-like
            If a scalar value is passed it is used to fill all missing values.
            Alternatively, an array-like 'value' can be given. It's expected
            that the array-like have the same length as 'self'.
        method : {'backfill', 'bfill', 'pad', 'ffill', None}, default None
            Method to use for filling holes in reindexed Series
            pad / ffill: propagate last valid observation forward to next valid
            backfill / bfill: use NEXT valid observation to fill gap.
        limit : int, default None
            If method is specified, this is the maximum number of consecutive
            NaN values to forward/backward fill. In other words, if there is
            a gap with more than this number of consecutive NaNs, it will only
            be partially filled. If method is not specified, this is the
            maximum number of entries along the entire axis where NaNs will be
            filled.

        Returns
        -------
        ExtensionArray
            With NA/NaN filled.
        """
        ...
    @property
    def nbytes(self) -> int:
        """
        The number of bytes needed to store this object in memory.
        """
        ...
    def isna(self) -> np.ndarray:
        """
        Boolean NumPy array indicating if each value is missing.

        This should return a 1-D array the same length as 'self'.
        """
        ...
    def copy(self) -> ArrowStringArray:
        """
        Return a shallow copy of the array.

        Returns
        -------
        ArrowStringArray
        """
        ...
    def __setitem__(self, key: int | slice | np.ndarray, value: Any) -> None:
        """Set one or more values inplace.

        Parameters
        ----------
        key : int, ndarray, or slice
            When called from, e.g. ``Series.__setitem__``, ``key`` will be
            one of

            * scalar int
            * ndarray of integers.
            * boolean ndarray
            * slice object

        value : ExtensionDtype.type, Sequence[ExtensionDtype.type], or object
            value or values to be set of ``key``.

        Returns
        -------
        None
        """
        ...
    def take(
        self, indices: Sequence[int], allow_fill: bool = ..., fill_value: Any = ...
    ):  # -> Self@ArrowStringArray:
        """
        Take elements from an array.

        Parameters
        ----------
        indices : sequence of int
            Indices to be taken.
        allow_fill : bool, default False
            How to handle negative values in `indices`.

            * False: negative values in `indices` indicate positional indices
              from the right (the default). This is similar to
              :func:`numpy.take`.

            * True: negative values in `indices` indicate
              missing values. These values are set to `fill_value`. Any other
              other negative values raise a ``ValueError``.

        fill_value : any, optional
            Fill value to use for NA-indices when `allow_fill` is True.
            This may be ``None``, in which case the default NA value for
            the type, ``self.dtype.na_value``, is used.

            For many ExtensionArrays, there will be two representations of
            `fill_value`: a user-facing "boxed" scalar, and a low-level
            physical NA value. `fill_value` should be the user-facing version,
            and the implementation should handle translating that to the
            physical version for processing the take if necessary.

        Returns
        -------
        ExtensionArray

        Raises
        ------
        IndexError
            When the indices are out of bounds for the array.
        ValueError
            When `indices` contains negative values other than ``-1``
            and `allow_fill` is True.

        See Also
        --------
        numpy.take
        api.extensions.take

        Notes
        -----
        ExtensionArray.take is called by ``Series.__getitem__``, ``.loc``,
        ``iloc``, when `indices` is a sequence of values. Additionally,
        it's called by :meth:`Series.reindex`, or any other method
        that causes realignment, with a `fill_value`.
        """
        ...
    def isin(self, values): ...
    def value_counts(self, dropna: bool = ...) -> Series:
        """
        Return a Series containing counts of each unique value.

        Parameters
        ----------
        dropna : bool, default True
            Don't include counts of missing values.

        Returns
        -------
        counts : Series

        See Also
        --------
        Series.value_counts
        """
        ...
    def astype(self, dtype, copy=...): ...
    _str_na_value = ...
