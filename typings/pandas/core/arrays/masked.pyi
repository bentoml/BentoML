from typing import TYPE_CHECKING, Any

import numpy as np
from pandas import Series
from pandas._typing import ArrayLike, Dtype, NpDtype, PositionalIndexer, Scalar, type_t
from pandas.core.arraylike import OpsMixin
from pandas.core.arrays import BooleanArray, ExtensionArray
from pandas.core.dtypes.base import ExtensionDtype
from pandas.util._decorators import cache_readonly, doc

if TYPE_CHECKING: ...
BaseMaskedArrayT = ...

class BaseMaskedDtype(ExtensionDtype):
    """
    Base class for dtypes for BasedMaskedArray subclasses.
    """

    name: str
    base = ...
    type: type
    na_value = ...
    @cache_readonly
    def numpy_dtype(self) -> np.dtype:
        """Return an instance of our numpy dtype"""
        ...
    @cache_readonly
    def kind(self) -> str: ...
    @cache_readonly
    def itemsize(self) -> int:
        """Return the number of bytes in this dtype"""
        ...
    @classmethod
    def construct_array_type(cls) -> type_t[BaseMaskedArray]:
        """
        Return the array type associated with this dtype.

        Returns
        -------
        type
        """
        ...

class BaseMaskedArray(OpsMixin, ExtensionArray):
    """
    Base class for masked arrays (which use _data and _mask to store the data).

    numpy based
    """

    _internal_fill_value: Scalar
    def __init__(
        self, values: np.ndarray, mask: np.ndarray, copy: bool = ...
    ) -> None: ...
    @property
    def dtype(self) -> BaseMaskedDtype: ...
    def __getitem__(self, item: PositionalIndexer) -> BaseMaskedArray | Any: ...
    @doc(ExtensionArray.fillna)
    def fillna(
        self: BaseMaskedArrayT, value=..., method=..., limit=...
    ) -> BaseMaskedArrayT: ...
    def __setitem__(self, key, value) -> None: ...
    def __iter__(self): ...
    def __len__(self) -> int: ...
    def __invert__(self: BaseMaskedArrayT) -> BaseMaskedArrayT: ...
    def to_numpy(
        self, dtype: NpDtype | None = ..., copy: bool = ..., na_value: Scalar = ...
    ) -> np.ndarray:
        """
        Convert to a NumPy Array.

        By default converts to an object-dtype NumPy array. Specify the `dtype` and
        `na_value` keywords to customize the conversion.

        Parameters
        ----------
        dtype : dtype, default object
            The numpy dtype to convert to.
        copy : bool, default False
            Whether to ensure that the returned value is a not a view on
            the array. Note that ``copy=False`` does not *ensure* that
            ``to_numpy()`` is no-copy. Rather, ``copy=True`` ensure that
            a copy is made, even if not strictly necessary. This is typically
            only possible when no missing values are present and `dtype`
            is the equivalent numpy dtype.
        na_value : scalar, optional
             Scalar missing value indicator to use in numpy array. Defaults
             to the native missing value indicator of this array (pd.NA).

        Returns
        -------
        numpy.ndarray

        Examples
        --------
        An object-dtype is the default result

        >>> a = pd.array([True, False, pd.NA], dtype="boolean")
        >>> a.to_numpy()
        array([True, False, <NA>], dtype=object)

        When no missing values are present, an equivalent dtype can be used.

        >>> pd.array([True, False], dtype="boolean").to_numpy(dtype="bool")
        array([ True, False])
        >>> pd.array([1, 2], dtype="Int64").to_numpy("int64")
        array([1, 2])

        However, requesting such dtype will raise a ValueError if
        missing values are present and the default missing value :attr:`NA`
        is used.

        >>> a = pd.array([True, False, pd.NA], dtype="boolean")
        >>> a
        <BooleanArray>
        [True, False, <NA>]
        Length: 3, dtype: boolean

        >>> a.to_numpy(dtype="bool")
        Traceback (most recent call last):
        ...
        ValueError: cannot convert to bool numpy array in presence of missing values

        Specify a valid `na_value` instead

        >>> a.to_numpy(dtype="bool", na_value=False)
        array([ True, False, False])
        """
        ...
    def astype(self, dtype: Dtype, copy: bool = ...) -> ArrayLike: ...
    __array_priority__ = ...
    def __array__(self, dtype: NpDtype | None = ...) -> np.ndarray:
        """
        the array interface, return my values
        We return an object array here to preserve our scalar values
        """
        ...
    def __arrow_array__(self, type=...):
        """
        Convert myself into a pyarrow Array.
        """
        ...
    def isna(self) -> np.ndarray: ...
    @property
    def nbytes(self) -> int: ...
    def take(
        self: BaseMaskedArrayT,
        indexer,
        *,
        allow_fill: bool = ...,
        fill_value: Scalar | None = ...
    ) -> BaseMaskedArrayT: ...
    def isin(self, values) -> BooleanArray: ...
    def copy(self: BaseMaskedArrayT) -> BaseMaskedArrayT: ...
    @doc(ExtensionArray.factorize)
    def factorize(self, na_sentinel: int = ...) -> tuple[np.ndarray, ExtensionArray]: ...
    def value_counts(self, dropna: bool = ...) -> Series:
        """
        Returns a Series containing counts of each unique value.

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
