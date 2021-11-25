from textwrap import dedent
from typing import TYPE_CHECKING, Literal

import numpy as np
from pandas import DataFrame, Index, Series
from pandas._typing import AnyArrayLike, ArrayLike, DtypeObj, FrameOrSeriesUnion, final
from pandas.util._decorators import doc

"""
Generic data algorithms. This module is experimental at the moment and not
intended for public consumption
"""
if TYPE_CHECKING: ...
_shared_docs: dict[str, str] = ...
_hashtables = ...

def get_data_algo(values: ArrayLike): ...
def unique(values):  # -> ArrayLike:
    """
    Hash table-based unique. Uniques are returned in order
    of appearance. This does NOT sort.

    Significantly faster than numpy.unique for long enough sequences.
    Includes NA values.

    Parameters
    ----------
    values : 1d array-like

    Returns
    -------
    numpy.ndarray or ExtensionArray

        The return can be:

        * Index : when the input is an Index
        * Categorical : when the input is a Categorical dtype
        * ndarray : when the input is a Series/ndarray

        Return numpy.ndarray or ExtensionArray.

    See Also
    --------
    Index.unique : Return unique values from an Index.
    Series.unique : Return unique values of Series object.

    Examples
    --------
    >>> pd.unique(pd.Series([2, 1, 3, 3]))
    array([2, 1, 3])

    >>> pd.unique(pd.Series([2] + [1] * 5))
    array([2, 1])

    >>> pd.unique(pd.Series([pd.Timestamp("20160101"), pd.Timestamp("20160101")]))
    array(['2016-01-01T00:00:00.000000000'], dtype='datetime64[ns]')

    >>> pd.unique(
    ...     pd.Series(
    ...         [
    ...             pd.Timestamp("20160101", tz="US/Eastern"),
    ...             pd.Timestamp("20160101", tz="US/Eastern"),
    ...         ]
    ...     )
    ... )
    <DatetimeArray>
    ['2016-01-01 00:00:00-05:00']
    Length: 1, dtype: datetime64[ns, US/Eastern]

    >>> pd.unique(
    ...     pd.Index(
    ...         [
    ...             pd.Timestamp("20160101", tz="US/Eastern"),
    ...             pd.Timestamp("20160101", tz="US/Eastern"),
    ...         ]
    ...     )
    ... )
    DatetimeIndex(['2016-01-01 00:00:00-05:00'],
            dtype='datetime64[ns, US/Eastern]',
            freq=None)

    >>> pd.unique(list("baabc"))
    array(['b', 'a', 'c'], dtype=object)

    An unordered Categorical will return categories in the
    order of appearance.

    >>> pd.unique(pd.Series(pd.Categorical(list("baabc"))))
    ['b', 'a', 'c']
    Categories (3, object): ['a', 'b', 'c']

    >>> pd.unique(pd.Series(pd.Categorical(list("baabc"), categories=list("abc"))))
    ['b', 'a', 'c']
    Categories (3, object): ['a', 'b', 'c']

    An ordered Categorical preserves the category ordering.

    >>> pd.unique(
    ...     pd.Series(
    ...         pd.Categorical(list("baabc"), categories=list("abc"), ordered=True)
    ...     )
    ... )
    ['b', 'a', 'c']
    Categories (3, object): ['a' < 'b' < 'c']

    An array of tuples

    >>> pd.unique([("a", "b"), ("b", "a"), ("a", "c"), ("b", "a")])
    array([('a', 'b'), ('b', 'a'), ('a', 'c')], dtype=object)
    """
    ...

unique1d = ...

def isin(comps: AnyArrayLike, values: AnyArrayLike) -> np.ndarray:
    """
    Compute the isin boolean array.

    Parameters
    ----------
    comps : array-like
    values : array-like

    Returns
    -------
    ndarray[bool]
        Same length as `comps`.
    """
    ...

def factorize_array(
    values: np.ndarray,
    na_sentinel: int = ...,
    size_hint: int | None = ...,
    na_value=...,
    mask: np.ndarray | None = ...,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Factorize an array-like to codes and uniques.

    This doesn't do any coercion of types or unboxing before factorization.

    Parameters
    ----------
    values : ndarray
    na_sentinel : int, default -1
    size_hint : int, optional
        Passed through to the hashtable's 'get_labels' method
    na_value : object, optional
        A value in `values` to consider missing. Note: only use this
        parameter when you know that you don't have any values pandas would
        consider missing in the array (NaN for float data, iNaT for
        datetimes, etc.).
    mask : ndarray[bool], optional
        If not None, the mask is used as indicator for missing values
        (True = missing, False = valid) instead of `na_value` or
        condition "val != val".

    Returns
    -------
    codes : ndarray[np.intp]
    uniques : ndarray
    """
    ...

@doc(
    values=dedent(
        """\
    values : sequence
        A 1-D sequence. Sequences that aren't pandas objects are
        coerced to ndarrays before factorization.
    """
    ),
    sort=dedent(
        """\
    sort : bool, default False
        Sort `uniques` and shuffle `codes` to maintain the
        relationship.
    """
    ),
    size_hint=dedent(
        """\
    size_hint : int, optional
        Hint to the hashtable sizer.
    """
    ),
)
def factorize(
    values, sort: bool = ..., na_sentinel: int | None = ..., size_hint: int | None = ...
) -> tuple[np.ndarray, np.ndarray | Index]:
    """
    Encode the object as an enumerated type or categorical variable.

    This method is useful for obtaining a numeric representation of an
    array when all that matters is identifying distinct values. `factorize`
    is available as both a top-level function :func:`pandas.factorize`,
    and as a method :meth:`Series.factorize` and :meth:`Index.factorize`.

    Parameters
    ----------
    {values}{sort}
    na_sentinel : int or None, default -1
        Value to mark "not found". If None, will not drop the NaN
        from the uniques of the values.

        .. versionchanged:: 1.1.2
    {size_hint}\

    Returns
    -------
    codes : ndarray
        An integer ndarray that's an indexer into `uniques`.
        ``uniques.take(codes)`` will have the same values as `values`.
    uniques : ndarray, Index, or Categorical
        The unique valid values. When `values` is Categorical, `uniques`
        is a Categorical. When `values` is some other pandas object, an
        `Index` is returned. Otherwise, a 1-D ndarray is returned.

        .. note ::

           Even if there's a missing value in `values`, `uniques` will
           *not* contain an entry for it.

    See Also
    --------
    cut : Discretize continuous-valued array.
    unique : Find the unique value in an array.

    Examples
    --------
    These examples all show factorize as a top-level method like
    ``pd.factorize(values)``. The results are identical for methods like
    :meth:`Series.factorize`.

    >>> codes, uniques = pd.factorize(['b', 'b', 'a', 'c', 'b'])
    >>> codes
    array([0, 0, 1, 2, 0]...)
    >>> uniques
    array(['b', 'a', 'c'], dtype=object)

    With ``sort=True``, the `uniques` will be sorted, and `codes` will be
    shuffled so that the relationship is the maintained.

    >>> codes, uniques = pd.factorize(['b', 'b', 'a', 'c', 'b'], sort=True)
    >>> codes
    array([1, 1, 0, 2, 1]...)
    >>> uniques
    array(['a', 'b', 'c'], dtype=object)

    Missing values are indicated in `codes` with `na_sentinel`
    (``-1`` by default). Note that missing values are never
    included in `uniques`.

    >>> codes, uniques = pd.factorize(['b', None, 'a', 'c', 'b'])
    >>> codes
    array([ 0, -1,  1,  2,  0]...)
    >>> uniques
    array(['b', 'a', 'c'], dtype=object)

    Thus far, we've only factorized lists (which are internally coerced to
    NumPy arrays). When factorizing pandas objects, the type of `uniques`
    will differ. For Categoricals, a `Categorical` is returned.

    >>> cat = pd.Categorical(['a', 'a', 'c'], categories=['a', 'b', 'c'])
    >>> codes, uniques = pd.factorize(cat)
    >>> codes
    array([0, 0, 1]...)
    >>> uniques
    ['a', 'c']
    Categories (3, object): ['a', 'b', 'c']

    Notice that ``'b'`` is in ``uniques.categories``, despite not being
    present in ``cat.values``.

    For all other pandas objects, an Index of the appropriate type is
    returned.

    >>> cat = pd.Series(['a', 'a', 'c'])
    >>> codes, uniques = pd.factorize(cat)
    >>> codes
    array([0, 0, 1]...)
    >>> uniques
    Index(['a', 'c'], dtype='object')

    If NaN is in the values, and we want to include NaN in the uniques of the
    values, it can be achieved by setting ``na_sentinel=None``.

    >>> values = np.array([1, 2, 1, np.nan])
    >>> codes, uniques = pd.factorize(values)  # default: na_sentinel=-1
    >>> codes
    array([ 0,  1,  0, -1])
    >>> uniques
    array([1., 2.])

    >>> codes, uniques = pd.factorize(values, na_sentinel=None)
    >>> codes
    array([0, 1, 0, 2])
    >>> uniques
    array([ 1.,  2., nan])
    """
    ...

def value_counts(
    values,
    sort: bool = ...,
    ascending: bool = ...,
    normalize: bool = ...,
    bins=...,
    dropna: bool = ...,
) -> Series:
    """
    Compute a histogram of the counts of non-null values.

    Parameters
    ----------
    values : ndarray (1-d)
    sort : bool, default True
        Sort by values
    ascending : bool, default False
        Sort in ascending order
    normalize: bool, default False
        If True then compute a relative histogram
    bins : integer, optional
        Rather than count values, group them into half-open bins,
        convenience for pd.cut, only works with numeric data
    dropna : bool, default True
        Don't include counts of NaN

    Returns
    -------
    Series
    """
    ...

def value_counts_arraylike(values, dropna: bool):  # -> tuple[ArrayLike, Any | ndarray]:
    """
    Parameters
    ----------
    values : arraylike
    dropna : bool

    Returns
    -------
    uniques : np.ndarray or ExtensionArray
    counts : np.ndarray
    """
    ...

def duplicated(
    values: ArrayLike, keep: Literal["first", "last", False] = ...
) -> np.ndarray:
    """
    Return boolean ndarray denoting duplicate values.

    Parameters
    ----------
    values : ndarray-like
        Array over which to check for duplicate values.
    keep : {'first', 'last', False}, default 'first'
        - ``first`` : Mark duplicates as ``True`` except for the first
          occurrence.
        - ``last`` : Mark duplicates as ``True`` except for the last
          occurrence.
        - False : Mark all duplicates as ``True``.

    Returns
    -------
    duplicated : ndarray[bool]
    """
    ...

def mode(values, dropna: bool = ...) -> Series:
    """
    Returns the mode(s) of an array.

    Parameters
    ----------
    values : array-like
        Array over which to check for duplicate values.
    dropna : bool, default True
        Don't consider counts of NaN/NaT.

    Returns
    -------
    mode : Series
    """
    ...

def rank(
    values: ArrayLike,
    axis: int = ...,
    method: str = ...,
    na_option: str = ...,
    ascending: bool = ...,
    pct: bool = ...,
) -> np.ndarray:
    """
    Rank the values along a given axis.

    Parameters
    ----------
    values : array-like
        Array whose values will be ranked. The number of dimensions in this
        array must not exceed 2.
    axis : int, default 0
        Axis over which to perform rankings.
    method : {'average', 'min', 'max', 'first', 'dense'}, default 'average'
        The method by which tiebreaks are broken during the ranking.
    na_option : {'keep', 'top'}, default 'keep'
        The method by which NaNs are placed in the ranking.
        - ``keep``: rank each NaN value with a NaN ranking
        - ``top``: replace each NaN with either +/- inf so that they
                   there are ranked at the top
    ascending : bool, default True
        Whether or not the elements should be ranked in ascending order.
    pct : bool, default False
        Whether or not to the display the returned rankings in integer form
        (e.g. 1, 2, 3) or in percentile form (e.g. 0.333..., 0.666..., 1).
    """
    ...

def checked_add_with_arr(
    arr: np.ndarray, b, arr_mask: np.ndarray | None = ..., b_mask: np.ndarray | None = ...
) -> np.ndarray:
    """
    Perform array addition that checks for underflow and overflow.

    Performs the addition of an int64 array and an int64 integer (or array)
    but checks that they do not result in overflow first. For elements that
    are indicated to be NaN, whether or not there is overflow for that element
    is automatically ignored.

    Parameters
    ----------
    arr : array addend.
    b : array or scalar addend.
    arr_mask : np.ndarray[bool] or None, default None
        array indicating which elements to exclude from checking
    b_mask : np.ndarray[bool] or None, default None
        array or scalar indicating which element(s) to exclude from checking

    Returns
    -------
    sum : An array for elements x + b for each element x in arr if b is
          a scalar or an array for elements x + y for each element pair
          (x, y) in (arr, b).

    Raises
    ------
    OverflowError if any x + y exceeds the maximum or minimum int64 value.
    """
    ...

def quantile(x, q, interpolation_method=...):  # -> float | Any | ndarray:
    """
    Compute sample quantile or quantiles of the input array. For example, q=0.5
    computes the median.

    The `interpolation_method` parameter supports three values, namely
    `fraction` (default), `lower` and `higher`. Interpolation is done only,
    if the desired quantile lies between two data points `i` and `j`. For
    `fraction`, the result is an interpolated value between `i` and `j`;
    for `lower`, the result is `i`, for `higher` the result is `j`.

    Parameters
    ----------
    x : ndarray
        Values from which to extract score.
    q : scalar or array
        Percentile at which to extract score.
    interpolation_method : {'fraction', 'lower', 'higher'}, optional
        This optional parameter specifies the interpolation method to use,
        when the desired quantile lies between two data points `i` and `j`:

        - fraction: `i + (j - i)*fraction`, where `fraction` is the
                    fractional part of the index surrounded by `i` and `j`.
        -lower: `i`.
        - higher: `j`.

    Returns
    -------
    score : float
        Score at percentile.

    Examples
    --------
    >>> from scipy import stats
    >>> a = np.arange(100)
    >>> stats.scoreatpercentile(a, 50)
    49.5

    """
    ...

class SelectN:
    def __init__(self, obj, n: int, keep: str) -> None: ...
    def compute(self, method: str) -> FrameOrSeriesUnion: ...
    @final
    def nlargest(self): ...
    @final
    def nsmallest(self): ...
    @final
    @staticmethod
    def is_valid_dtype_n_method(dtype: DtypeObj) -> bool:
        """
        Helper function to determine if dtype is valid for
        nsmallest/nlargest methods
        """
        ...

class SelectNSeries(SelectN):
    """
    Implement n largest/smallest for Series

    Parameters
    ----------
    obj : Series
    n : int
    keep : {'first', 'last'}, default 'first'

    Returns
    -------
    nordered : Series
    """

    def compute(self, method: str) -> Series: ...

class SelectNFrame(SelectN):
    """
    Implement n largest/smallest for DataFrame

    Parameters
    ----------
    obj : DataFrame
    n : int
    keep : {'first', 'last'}, default 'first'
    columns : list or str

    Returns
    -------
    nordered : DataFrame
    """

    def __init__(self, obj, n: int, keep: str, columns) -> None: ...
    def compute(self, method: str) -> DataFrame: ...

def take(
    arr, indices: np.ndarray, axis: int = ..., allow_fill: bool = ..., fill_value=...
):  # -> ndarray:
    """
    Take elements from an array.

    Parameters
    ----------
    arr : sequence
        Non array-likes (sequences without a dtype) are coerced
        to an ndarray.
    indices : sequence of integers
        Indices to be taken.
    axis : int, default 0
        The axis over which to select values.
    allow_fill : bool, default False
        How to handle negative values in `indices`.

        * False: negative values in `indices` indicate positional indices
          from the right (the default). This is similar to :func:`numpy.take`.

        * True: negative values in `indices` indicate
          missing values. These values are set to `fill_value`. Any other
          negative values raise a ``ValueError``.

    fill_value : any, optional
        Fill value to use for NA-indices when `allow_fill` is True.
        This may be ``None``, in which case the default NA value for
        the type (``self.dtype.na_value``) is used.

        For multi-dimensional `arr`, each *element* is filled with
        `fill_value`.

    Returns
    -------
    ndarray or ExtensionArray
        Same type as the input.

    Raises
    ------
    IndexError
        When `indices` is out of bounds for the array.
    ValueError
        When the indexer contains negative values other than ``-1``
        and `allow_fill` is True.

    Notes
    -----
    When `allow_fill` is False, `indices` may be whatever dimensionality
    is accepted by NumPy for `arr`.

    When `allow_fill` is True, `indices` should be 1-D.

    See Also
    --------
    numpy.take : Take elements from an array along an axis.

    Examples
    --------
    >>> from pandas.api.extensions import take

    With the default ``allow_fill=False``, negative numbers indicate
    positional indices from the right.

    >>> take(np.array([10, 20, 30]), [0, 0, -1])
    array([10, 10, 30])

    Setting ``allow_fill=True`` will place `fill_value` in those positions.

    >>> take(np.array([10, 20, 30]), [0, 0, -1], allow_fill=True)
    array([10., 10., nan])

    >>> take(np.array([10, 20, 30]), [0, 0, -1], allow_fill=True,
    ...      fill_value=-10)
    array([ 10,  10, -10])
    """
    ...

def searchsorted(arr, value, side=..., sorter=...) -> np.ndarray:
    """
    Find indices where elements should be inserted to maintain order.

    .. versionadded:: 0.25.0

    Find the indices into a sorted array `arr` (a) such that, if the
    corresponding elements in `value` were inserted before the indices,
    the order of `arr` would be preserved.

    Assuming that `arr` is sorted:

    ======  ================================
    `side`  returned index `i` satisfies
    ======  ================================
    left    ``arr[i-1] < value <= self[i]``
    right   ``arr[i-1] <= value < self[i]``
    ======  ================================

    Parameters
    ----------
    arr: array-like
        Input array. If `sorter` is None, then it must be sorted in
        ascending order, otherwise `sorter` must be an array of indices
        that sort it.
    value : array-like
        Values to insert into `arr`.
    side : {'left', 'right'}, optional
        If 'left', the index of the first suitable location found is given.
        If 'right', return the last such index.  If there is no suitable
        index, return either 0 or N (where N is the length of `self`).
    sorter : 1-D array-like, optional
        Optional array of integer indices that sort array a into ascending
        order. They are typically the result of argsort.

    Returns
    -------
    array of ints
        Array of insertion points with the same shape as `value`.

    See Also
    --------
    numpy.searchsorted : Similar method from NumPy.
    """
    ...

_diff_special = ...

def diff(arr, n: int, axis: int = ..., stacklevel: int = ...):  # -> Any | ndarray:
    """
    difference of n between self,
    analogous to s-s.shift(n)

    Parameters
    ----------
    arr : ndarray or ExtensionArray
    n : int
        number of periods
    axis : {0, 1}
        axis to shift on
    stacklevel : int, default 3
        The stacklevel for the lost dtype warning.

    Returns
    -------
    shifted
    """
    ...

def safe_sort(
    values,
    codes=...,
    na_sentinel: int = ...,
    assume_unique: bool = ...,
    verify: bool = ...,
) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
    """
    Sort ``values`` and reorder corresponding ``codes``.

    ``values`` should be unique if ``codes`` is not None.
    Safe for use with mixed types (int, str), orders ints before strs.

    Parameters
    ----------
    values : list-like
        Sequence; must be unique if ``codes`` is not None.
    codes : list_like, optional
        Indices to ``values``. All out of bound indices are treated as
        "not found" and will be masked with ``na_sentinel``.
    na_sentinel : int, default -1
        Value in ``codes`` to mark "not found".
        Ignored when ``codes`` is None.
    assume_unique : bool, default False
        When True, ``values`` are assumed to be unique, which can speed up
        the calculation. Ignored when ``codes`` is None.
    verify : bool, default True
        Check if codes are out of bound for the values and put out of bound
        codes equal to na_sentinel. If ``verify=False``, it is assumed there
        are no out of bound codes. Ignored when ``codes`` is None.

        .. versionadded:: 0.25.0

    Returns
    -------
    ordered : ndarray
        Sorted ``values``
    new_codes : ndarray
        Reordered ``codes``; returned when ``codes`` is not None.

    Raises
    ------
    TypeError
        * If ``values`` is not list-like or if ``codes`` is neither None
        nor list-like
        * If ``values`` cannot be sorted
    ValueError
        * If ``codes`` is not None and ``values`` contain duplicates.
    """
    ...

def union_with_duplicates(lvals: ArrayLike, rvals: ArrayLike) -> ArrayLike:
    """
    Extracts the union from lvals and rvals with respect to duplicates and nans in
    both arrays.

    Parameters
    ----------
    lvals: np.ndarray or ExtensionArray
        left values which is ordered in front.
    rvals: np.ndarray or ExtensionArray
        right values ordered after lvals.

    Returns
    -------
    np.ndarray or ExtensionArray
        Containing the unsorted union of both arrays.
    """
    ...
