import numpy as np
from pandas._libs.sparse import SparseIndex
from pandas._typing import Dtype, NpDtype, Scalar
from pandas.core.arraylike import OpsMixin
from pandas.core.arrays import ExtensionArray
from pandas.core.arrays.sparse.dtype import SparseDtype
from pandas.core.base import PandasObject

"""
SparseArray data structure
"""
SparseArrayT = ...
_sparray_doc_kwargs = ...

class SparseArray(OpsMixin, PandasObject, ExtensionArray):
    """
    An ExtensionArray for storing sparse data.

    Parameters
    ----------
    data : array-like
        A dense array of values to store in the SparseArray. This may contain
        `fill_value`.
    sparse_index : SparseIndex, optional
    index : Index
    fill_value : scalar, optional
        Elements in `data` that are `fill_value` are not stored in the
        SparseArray. For memory savings, this should be the most common value
        in `data`. By default, `fill_value` depends on the dtype of `data`:

        =========== ==========
        data.dtype  na_value
        =========== ==========
        float       ``np.nan``
        int         ``0``
        bool        False
        datetime64  ``pd.NaT``
        timedelta64 ``pd.NaT``
        =========== ==========

        The fill value is potentially specified in three ways. In order of
        precedence, these are

        1. The `fill_value` argument
        2. ``dtype.fill_value`` if `fill_value` is None and `dtype` is
           a ``SparseDtype``
        3. ``data.dtype.fill_value`` if `fill_value` is None and `dtype`
           is not a ``SparseDtype`` and `data` is a ``SparseArray``.

    kind : {'integer', 'block'}, default 'integer'
        The type of storage for sparse locations.

        * 'block': Stores a `block` and `block_length` for each
          contiguous *span* of sparse values. This is best when
          sparse data tends to be clumped together, with large
          regions of ``fill-value`` values between sparse values.
        * 'integer': uses an integer to store the location of
          each sparse value.

    dtype : np.dtype or SparseDtype, optional
        The dtype to use for the SparseArray. For numpy dtypes, this
        determines the dtype of ``self.sp_values``. For SparseDtype,
        this determines ``self.sp_values`` and ``self.fill_value``.
    copy : bool, default False
        Whether to explicitly copy the incoming `data` array.

    Attributes
    ----------
    None

    Methods
    -------
    None

    Examples
    --------
    >>> from pandas.arrays import SparseArray
    >>> arr = SparseArray([0, 0, 1, 2])
    >>> arr
    [0, 0, 1, 2]
    Fill: 0
    IntIndex
    Indices: array([2, 3], dtype=int32)
    """

    _subtyp = ...
    _hidden_attrs = ...
    _sparse_index: SparseIndex
    def __init__(
        self,
        data,
        sparse_index=...,
        index=...,
        fill_value=...,
        kind=...,
        dtype: Dtype | None = ...,
        copy=...,
    ) -> None: ...
    @classmethod
    def from_spmatrix(cls, data):  # -> Self@SparseArray:
        """
        Create a SparseArray from a scipy.sparse matrix.

        .. versionadded:: 0.25.0

        Parameters
        ----------
        data : scipy.sparse.sp_matrix
            This should be a SciPy sparse matrix where the size
            of the second dimension is 1. In other words, a
            sparse matrix with a single column.

        Returns
        -------
        SparseArray

        Examples
        --------
        >>> import scipy.sparse
        >>> mat = scipy.sparse.coo_matrix((4, 1))
        >>> pd.arrays.SparseArray.from_spmatrix(mat)
        [0.0, 0.0, 0.0, 0.0]
        Fill: 0.0
        IntIndex
        Indices: array([], dtype=int32)
        """
        ...
    def __array__(self, dtype: NpDtype | None = ...) -> np.ndarray: ...
    def __setitem__(self, key, value): ...
    @property
    def sp_index(self) -> SparseIndex:
        """
        The SparseIndex containing the location of non- ``fill_value`` points.
        """
        ...
    @property
    def sp_values(self) -> np.ndarray:
        """
        An ndarray containing the non- ``fill_value`` values.

        Examples
        --------
        >>> s = SparseArray([0, 0, 1, 0, 2], fill_value=0)
        >>> s.sp_values
        array([1, 2])
        """
        ...
    @property
    def dtype(self) -> SparseDtype: ...
    @property
    def fill_value(self):  # -> Any:
        """
        Elements in `data` that are `fill_value` are not stored.

        For memory savings, this should be the most common value in the array.
        """
        ...
    @fill_value.setter
    def fill_value(self, value): ...
    @property
    def kind(self) -> str:
        """
        The kind of sparse index for this array. One of {'integer', 'block'}.
        """
        ...
    def __len__(self) -> int: ...
    @property
    def nbytes(self) -> int: ...
    @property
    def density(self) -> float:
        """
        The percent of non- ``fill_value`` points, as decimal.

        Examples
        --------
        >>> s = SparseArray([0, 0, 1, 1, 1], fill_value=0)
        >>> s.density
        0.6
        """
        ...
    @property
    def npoints(self) -> int:
        """
        The number of non- ``fill_value`` points.

        Examples
        --------
        >>> s = SparseArray([0, 0, 1, 1, 1], fill_value=0)
        >>> s.npoints
        3
        """
        ...
    def isna(self): ...
    def fillna(self, value=..., method=..., limit=...):  # -> Self@SparseArray:
        """
        Fill missing values with `value`.

        Parameters
        ----------
        value : scalar, optional
        method : str, optional

            .. warning::

               Using 'method' will result in high memory use,
               as all `fill_value` methods will be converted to
               an in-memory ndarray

        limit : int, optional

        Returns
        -------
        SparseArray

        Notes
        -----
        When `value` is specified, the result's ``fill_value`` depends on
        ``self.fill_value``. The goal is to maintain low-memory use.

        If ``self.fill_value`` is NA, the result dtype will be
        ``SparseDtype(self.dtype, fill_value=value)``. This will preserve
        amount of memory used before and after filling.

        When ``self.fill_value`` is not NA, the result dtype will be
        ``self.dtype``. Again, this preserves the amount of memory used.
        """
        ...
    def shift(self, periods=..., fill_value=...): ...
    def unique(self): ...
    def factorize(self, na_sentinel=...): ...
    def value_counts(self, dropna: bool = ...):  # -> Series:
        """
        Returns a Series containing counts of unique values.

        Parameters
        ----------
        dropna : bool, default True
            Don't include counts of NaN, even if NaN is in sp_values.

        Returns
        -------
        counts : Series
        """
        ...
    def __getitem__(self, key): ...
    def take(self, indices, *, allow_fill=..., fill_value=...) -> SparseArray: ...
    def searchsorted(self, v, side=..., sorter=...): ...
    def copy(self: SparseArrayT) -> SparseArrayT: ...
    def astype(self, dtype: Dtype | None = ..., copy=...):  # -> Self@SparseArray:
        """
        Change the dtype of a SparseArray.

        The output will always be a SparseArray. To convert to a dense
        ndarray with a certain dtype, use :meth:`numpy.asarray`.

        Parameters
        ----------
        dtype : np.dtype or ExtensionDtype
            For SparseDtype, this changes the dtype of
            ``self.sp_values`` and the ``self.fill_value``.

            For other dtypes, this only changes the dtype of
            ``self.sp_values``.

        copy : bool, default True
            Whether to ensure a copy is made, even if not necessary.

        Returns
        -------
        SparseArray

        Examples
        --------
        >>> arr = pd.arrays.SparseArray([0, 0, 1, 2])
        >>> arr
        [0, 0, 1, 2]
        Fill: 0
        IntIndex
        Indices: array([2, 3], dtype=int32)

        >>> arr.astype(np.dtype('int32'))
        [0, 0, 1, 2]
        Fill: 0
        IntIndex
        Indices: array([2, 3], dtype=int32)

        Using a NumPy dtype with a different kind (e.g. float) will coerce
        just ``self.sp_values``.

        >>> arr.astype(np.dtype('float64'))
        ... # doctest: +NORMALIZE_WHITESPACE
        [0.0, 0.0, 1.0, 2.0]
        Fill: 0.0
        IntIndex
        Indices: array([2, 3], dtype=int32)

        Use a SparseDtype if you wish to be change the fill value as well.

        >>> arr.astype(SparseDtype("float64", fill_value=np.nan))
        ... # doctest: +NORMALIZE_WHITESPACE
        [nan, nan, 1.0, 2.0]
        Fill: nan
        IntIndex
        Indices: array([2, 3], dtype=int32)
        """
        ...
    def map(self, mapper):  # -> Self@SparseArray:
        """
        Map categories using input correspondence (dict, Series, or function).

        Parameters
        ----------
        mapper : dict, Series, callable
            The correspondence from old values to new.

        Returns
        -------
        SparseArray
            The output array will have the same density as the input.
            The output fill value will be the result of applying the
            mapping to ``self.fill_value``

        Examples
        --------
        >>> arr = pd.arrays.SparseArray([0, 1, 2])
        >>> arr.map(lambda x: x + 10)
        [10, 11, 12]
        Fill: 10
        IntIndex
        Indices: array([1, 2], dtype=int32)

        >>> arr.map({0: 10, 1: 11, 2: 12})
        [10, 11, 12]
        Fill: 10
        IntIndex
        Indices: array([1, 2], dtype=int32)

        >>> arr.map(pd.Series([10, 11, 12], index=[0, 1, 2]))
        [10, 11, 12]
        Fill: 10
        IntIndex
        Indices: array([1, 2], dtype=int32)
        """
        ...
    def to_dense(self):  # -> ndarray:
        """
        Convert SparseArray to a NumPy array.

        Returns
        -------
        arr : NumPy array
        """
        ...
    _internal_get_values = ...
    def __setstate__(self, state):  # -> None:
        """Necessary for making this object picklable"""
        ...
    def nonzero(self): ...
    def all(self, axis=..., *args, **kwargs):  # -> bool_ | Literal[False]:
        """
        Tests whether all elements evaluate True

        Returns
        -------
        all : bool

        See Also
        --------
        numpy.all
        """
        ...
    def any(self, axis=..., *args, **kwargs):  # -> bool:
        """
        Tests whether at least one of elements evaluate True

        Returns
        -------
        any : bool

        See Also
        --------
        numpy.any
        """
        ...
    def sum(self, axis: int = ..., min_count: int = ..., *args, **kwargs) -> Scalar:
        """
        Sum of non-NA/null values

        Parameters
        ----------
        axis : int, default 0
            Not Used. NumPy compatibility.
        min_count : int, default 0
            The required number of valid values to perform the summation. If fewer
            than ``min_count`` valid values are present, the result will be the missing
            value indicator for subarray type.
        *args, **kwargs
            Not Used. NumPy compatibility.

        Returns
        -------
        scalar
        """
        ...
    def cumsum(self, axis=..., *args, **kwargs):  # -> SparseArray:
        """
        Cumulative sum of non-NA/null values.

        When performing the cumulative summation, any non-NA/null values will
        be skipped. The resulting SparseArray will preserve the locations of
        NaN values, but the fill value will be `np.nan` regardless.

        Parameters
        ----------
        axis : int or None
            Axis over which to perform the cumulative summation. If None,
            perform cumulative summation over flattened array.

        Returns
        -------
        cumsum : SparseArray
        """
        ...
    def mean(self, axis=..., *args, **kwargs):  # -> Any:
        """
        Mean of non-NA/null values

        Returns
        -------
        mean : float
        """
        ...
    def max(self, axis=..., *args, **kwargs): ...
    def min(self, axis=..., *args, **kwargs): ...
    _HANDLED_TYPES = ...
    def __array_ufunc__(self, ufunc: np.ufunc, method: str, *inputs, **kwargs): ...
    def __abs__(self): ...
    _logical_method = ...
    def __pos__(self) -> SparseArray: ...
    def __neg__(self) -> SparseArray: ...
    def __invert__(self) -> SparseArray: ...
    def __repr__(self) -> str: ...

def make_sparse(
    arr: np.ndarray, kind=..., fill_value=..., dtype: NpDtype | None = ...
):  # -> tuple[Unknown | Any, Unknown, object | Unknown | float | Literal[0, False]]:
    """
    Convert ndarray to sparse format

    Parameters
    ----------
    arr : ndarray
    kind : {'block', 'integer'}
    fill_value : NaN or another value
    dtype : np.dtype, optional
    copy : bool, default False

    Returns
    -------
    (sparse_values, index, fill_value) : (ndarray, SparseIndex, Scalar)
    """
    ...

def make_sparse_index(length, indices, kind): ...
