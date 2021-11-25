from typing import TYPE_CHECKING, Any, Sequence

import numpy as np
import numpy.ma as ma
from pandas import ExtensionArray, Index, Series
from pandas._typing import AnyArrayLike, ArrayLike, Dtype, DtypeObj

"""
Constructor functions intended to be shared by pd.array, Series.__init__,
and Index.__new__.

These should not depend on core.internals.
"""
if TYPE_CHECKING: ...

def array(
    data: Sequence[object] | AnyArrayLike, dtype: Dtype | None = ..., copy: bool = ...
) -> ExtensionArray:
    """
    Create an array.

    Parameters
    ----------
    data : Sequence of objects
        The scalars inside `data` should be instances of the
        scalar type for `dtype`. It's expected that `data`
        represents a 1-dimensional array of data.

        When `data` is an Index or Series, the underlying array
        will be extracted from `data`.

    dtype : str, np.dtype, or ExtensionDtype, optional
        The dtype to use for the array. This may be a NumPy
        dtype or an extension type registered with pandas using
        :meth:`pandas.api.extensions.register_extension_dtype`.

        If not specified, there are two possibilities:

        1. When `data` is a :class:`Series`, :class:`Index`, or
           :class:`ExtensionArray`, the `dtype` will be taken
           from the data.
        2. Otherwise, pandas will attempt to infer the `dtype`
           from the data.

        Note that when `data` is a NumPy array, ``data.dtype`` is
        *not* used for inferring the array type. This is because
        NumPy cannot represent all the types of data that can be
        held in extension arrays.

        Currently, pandas will infer an extension dtype for sequences of

        ============================== =======================================
        Scalar Type                    Array Type
        ============================== =======================================
        :class:`pandas.Interval`       :class:`pandas.arrays.IntervalArray`
        :class:`pandas.Period`         :class:`pandas.arrays.PeriodArray`
        :class:`datetime.datetime`     :class:`pandas.arrays.DatetimeArray`
        :class:`datetime.timedelta`    :class:`pandas.arrays.TimedeltaArray`
        :class:`int`                   :class:`pandas.arrays.IntegerArray`
        :class:`float`                 :class:`pandas.arrays.FloatingArray`
        :class:`str`                   :class:`pandas.arrays.StringArray` or
                                       :class:`pandas.arrays.ArrowStringArray`
        :class:`bool`                  :class:`pandas.arrays.BooleanArray`
        ============================== =======================================

        The ExtensionArray created when the scalar type is :class:`str` is determined by
        ``pd.options.mode.string_storage`` if the dtype is not explicitly given.

        For all other cases, NumPy's usual inference rules will be used.

        .. versionchanged:: 1.0.0

           Pandas infers nullable-integer dtype for integer data,
           string dtype for string data, and nullable-boolean dtype
           for boolean data.

        .. versionchanged:: 1.2.0

            Pandas now also infers nullable-floating dtype for float-like
            input data

    copy : bool, default True
        Whether to copy the data, even if not necessary. Depending
        on the type of `data`, creating the new array may require
        copying data, even if ``copy=False``.

    Returns
    -------
    ExtensionArray
        The newly created array.

    Raises
    ------
    ValueError
        When `data` is not 1-dimensional.

    See Also
    --------
    numpy.array : Construct a NumPy array.
    Series : Construct a pandas Series.
    Index : Construct a pandas Index.
    arrays.PandasArray : ExtensionArray wrapping a NumPy array.
    Series.array : Extract the array stored within a Series.

    Notes
    -----
    Omitting the `dtype` argument means pandas will attempt to infer the
    best array type from the values in the data. As new array types are
    added by pandas and 3rd party libraries, the "best" array type may
    change. We recommend specifying `dtype` to ensure that

    1. the correct array type for the data is returned
    2. the returned array type doesn't change as new extension types
       are added by pandas and third-party libraries

    Additionally, if the underlying memory representation of the returned
    array matters, we recommend specifying the `dtype` as a concrete object
    rather than a string alias or allowing it to be inferred. For example,
    a future version of pandas or a 3rd-party library may include a
    dedicated ExtensionArray for string data. In this event, the following
    would no longer return a :class:`arrays.PandasArray` backed by a NumPy
    array.

    >>> pd.array(['a', 'b'], dtype=str)
    <PandasArray>
    ['a', 'b']
    Length: 2, dtype: str32

    This would instead return the new ExtensionArray dedicated for string
    data. If you really need the new array to be backed by a  NumPy array,
    specify that in the dtype.

    >>> pd.array(['a', 'b'], dtype=np.dtype("<U1"))
    <PandasArray>
    ['a', 'b']
    Length: 2, dtype: str32

    Finally, Pandas has arrays that mostly overlap with NumPy

      * :class:`arrays.DatetimeArray`
      * :class:`arrays.TimedeltaArray`

    When data with a ``datetime64[ns]`` or ``timedelta64[ns]`` dtype is
    passed, pandas will always return a ``DatetimeArray`` or ``TimedeltaArray``
    rather than a ``PandasArray``. This is for symmetry with the case of
    timezone-aware data, which NumPy does not natively support.

    >>> pd.array(['2015', '2016'], dtype='datetime64[ns]')
    <DatetimeArray>
    ['2015-01-01 00:00:00', '2016-01-01 00:00:00']
    Length: 2, dtype: datetime64[ns]

    >>> pd.array(["1H", "2H"], dtype='timedelta64[ns]')
    <TimedeltaArray>
    ['0 days 01:00:00', '0 days 02:00:00']
    Length: 2, dtype: timedelta64[ns]

    Examples
    --------
    If a dtype is not specified, pandas will infer the best dtype from the values.
    See the description of `dtype` for the types pandas infers for.

    >>> pd.array([1, 2])
    <IntegerArray>
    [1, 2]
    Length: 2, dtype: Int64

    >>> pd.array([1, 2, np.nan])
    <IntegerArray>
    [1, 2, <NA>]
    Length: 3, dtype: Int64

    >>> pd.array([1.1, 2.2])
    <FloatingArray>
    [1.1, 2.2]
    Length: 2, dtype: Float64

    >>> pd.array(["a", None, "c"])
    <StringArray>
    ['a', <NA>, 'c']
    Length: 3, dtype: string

    >>> with pd.option_context("string_storage", "pyarrow"):
    ...     arr = pd.array(["a", None, "c"])
    ...
    >>> arr
    <ArrowStringArray>
    ['a', <NA>, 'c']
    Length: 3, dtype: string

    >>> pd.array([pd.Period('2000', freq="D"), pd.Period("2000", freq="D")])
    <PeriodArray>
    ['2000-01-01', '2000-01-01']
    Length: 2, dtype: period[D]

    You can use the string alias for `dtype`

    >>> pd.array(['a', 'b', 'a'], dtype='category')
    ['a', 'b', 'a']
    Categories (2, object): ['a', 'b']

    Or specify the actual dtype

    >>> pd.array(['a', 'b', 'a'],
    ...          dtype=pd.CategoricalDtype(['a', 'b', 'c'], ordered=True))
    ['a', 'b', 'a']
    Categories (3, object): ['a' < 'b' < 'c']

    If pandas does not infer a dedicated extension type a
    :class:`arrays.PandasArray` is returned.

    >>> pd.array([1 + 1j, 3 + 2j])
    <PandasArray>
    [(1+1j), (3+2j)]
    Length: 2, dtype: complex128

    As mentioned in the "Notes" section, new extension types may be added
    in the future (by pandas or 3rd party libraries), causing the return
    value to no longer be a :class:`arrays.PandasArray`. Specify the `dtype`
    as a NumPy dtype if you need to ensure there's no future change in
    behavior.

    >>> pd.array([1, 2], dtype=np.dtype("int32"))
    <PandasArray>
    [1, 2]
    Length: 2, dtype: int32

    `data` must be 1-dimensional. A ValueError is raised when the input
    has the wrong dimensionality.

    >>> pd.array(1)
    Traceback (most recent call last):
      ...
    ValueError: Cannot pass scalar '1' to 'pandas.array'.
    """
    ...

def extract_array(
    obj: object, extract_numpy: bool = ..., extract_range: bool = ...
) -> Any | ArrayLike:
    """
    Extract the ndarray or ExtensionArray from a Series or Index.

    For all other types, `obj` is just returned as is.

    Parameters
    ----------
    obj : object
        For Series / Index, the underlying ExtensionArray is unboxed.
        For Numpy-backed ExtensionArrays, the ndarray is extracted.

    extract_numpy : bool, default False
        Whether to extract the ndarray from a PandasArray

    extract_range : bool, default False
        If we have a RangeIndex, return range._values if True
        (which is a materialized integer ndarray), otherwise return unchanged.

    Returns
    -------
    arr : object

    Examples
    --------
    >>> extract_array(pd.Series(['a', 'b', 'c'], dtype='category'))
    ['a', 'b', 'c']
    Categories (3, object): ['a', 'b', 'c']

    Other objects like lists, arrays, and DataFrames are just passed through.

    >>> extract_array([1, 2, 3])
    [1, 2, 3]

    For an ndarray-backed Series / Index a PandasArray is returned.

    >>> extract_array(pd.Series([1, 2, 3]))
    <PandasArray>
    [1, 2, 3]
    Length: 3, dtype: int64

    To extract all the way down to the ndarray, pass ``extract_numpy=True``.

    >>> extract_array(pd.Series([1, 2, 3]), extract_numpy=True)
    array([1, 2, 3])
    """
    ...

def ensure_wrapped_if_datetimelike(arr):  # -> DatetimeArray | TimedeltaArray | ndarray:
    """
    Wrap datetime64 and timedelta64 ndarrays in DatetimeArray/TimedeltaArray.
    """
    ...

def sanitize_masked_array(data: ma.MaskedArray) -> np.ndarray:
    """
    Convert numpy MaskedArray to ensure mask is softened.
    """
    ...

def sanitize_array(
    data,
    index: Index | None,
    dtype: DtypeObj | None = ...,
    copy: bool = ...,
    raise_cast_failure: bool = ...,
    *,
    allow_2d: bool = ...
) -> ArrayLike:
    """
    Sanitize input data to an ndarray or ExtensionArray, copy if specified,
    coerce to the dtype if specified.

    Parameters
    ----------
    data : Any
    index : Index or None, default None
    dtype : np.dtype, ExtensionDtype, or None, default None
    copy : bool, default False
    raise_cast_failure : bool, default True
    allow_2d : bool, default False
        If False, raise if we have a 2D Arraylike.

    Returns
    -------
    np.ndarray or ExtensionArray

    Notes
    -----
    raise_cast_failure=False is only intended to be True when called from the
    DataFrame constructor, as the dtype keyword there may be interpreted as only
    applying to a subset of columns, see GH#24435.
    """
    ...

def range_to_ndarray(rng: range) -> np.ndarray:
    """
    Cast a range object to ndarray.
    """
    ...

def is_empty_data(data: Any) -> bool:
    """
    Utility to check if a Series is instantiated with empty data,
    which does not contain dtype information.

    Parameters
    ----------
    data : array-like, Iterable, dict, or scalar value
        Contains data stored in Series.

    Returns
    -------
    bool
    """
    ...

def create_series_with_explicit_dtype(
    data: Any = ...,
    index: ArrayLike | Index | None = ...,
    dtype: Dtype | None = ...,
    name: str | None = ...,
    copy: bool = ...,
    fastpath: bool = ...,
    dtype_if_empty: Dtype = ...,
) -> Series:
    """
    Helper to pass an explicit dtype when instantiating an empty Series.

    This silences a DeprecationWarning described in GitHub-17261.

    Parameters
    ----------
    data : Mirrored from Series.__init__
    index : Mirrored from Series.__init__
    dtype : Mirrored from Series.__init__
    name : Mirrored from Series.__init__
    copy : Mirrored from Series.__init__
    fastpath : Mirrored from Series.__init__
    dtype_if_empty : str, numpy.dtype, or ExtensionDtype
        This dtype will be passed explicitly if an empty Series will
        be instantiated.

    Returns
    -------
    Series
    """
    ...
