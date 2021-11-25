from typing import TYPE_CHECKING, Any, Sized, overload

import numpy as np
from pandas._typing import ArrayLike, Dtype, DtypeObj, Scalar
from pandas.core.arrays import (
    DatetimeArray,
    ExtensionArray,
    IntervalArray,
    PeriodArray,
    TimedeltaArray,
)
from pandas.core.dtypes.dtypes import ExtensionDtype

"""
Routines for casting.
"""
if TYPE_CHECKING: ...
_int8_max = np.iinfo(np.int8).max
_int16_max = np.iinfo(np.int16).max
_int32_max = np.iinfo(np.int32).max
_int64_max = np.iinfo(np.int64).max
NumpyArrayT = ...

def maybe_convert_platform(
    values: list | tuple | range | np.ndarray | ExtensionArray,
) -> ArrayLike:
    """try to do platform conversion, allow ndarray or list here"""
    ...

def is_nested_object(obj) -> bool:
    """
    return a boolean if we have a nested object, e.g. a Series with 1 or
    more Series elements

    This may not be necessarily be performant.

    """
    ...

def maybe_box_datetimelike(value: Scalar, dtype: Dtype | None = ...) -> Scalar:
    """
    Cast scalar to Timestamp or Timedelta if scalar is datetime-like
    and dtype is not object.

    Parameters
    ----------
    value : scalar
    dtype : Dtype, optional

    Returns
    -------
    scalar
    """
    ...

def maybe_box_native(value: Scalar) -> Scalar:
    """
    If passed a scalar cast the scalar to a python native type.

    Parameters
    ----------
    value : scalar or Series

    Returns
    -------
    scalar or Series
    """
    ...

def maybe_unbox_datetimelike(value: Scalar, dtype: DtypeObj) -> Scalar:
    """
    Convert a Timedelta or Timestamp to timedelta64 or datetime64 for setting
    into a numpy array.  Failing to unbox would risk dropping nanoseconds.

    Notes
    -----
    Caller is responsible for checking dtype.kind in ["m", "M"]
    """
    ...

def maybe_downcast_to_dtype(result: ArrayLike, dtype: str | np.dtype) -> ArrayLike:
    """
    try to cast to the specified dtype (e.g. convert back to bool/int
    or could be an astype of float64->float32
    """
    ...

def maybe_downcast_numeric(
    result: ArrayLike, dtype: DtypeObj, do_round: bool = ...
) -> ArrayLike:
    """
    Subset of maybe_downcast_to_dtype restricted to numeric dtypes.

    Parameters
    ----------
    result : ndarray or ExtensionArray
    dtype : np.dtype or ExtensionDtype
    do_round : bool

    Returns
    -------
    ndarray or ExtensionArray
    """
    ...

def maybe_cast_pointwise_result(
    result: ArrayLike, dtype: DtypeObj, numeric_only: bool = ..., same_dtype: bool = ...
) -> ArrayLike:
    """
    Try casting result of a pointwise operation back to the original dtype if
    appropriate.

    Parameters
    ----------
    result : array-like
        Result to cast.
    dtype : np.dtype or ExtensionDtype
        Input Series from which result was calculated.
    numeric_only : bool, default False
        Whether to cast only numerics or datetimes as well.
    same_dtype : bool, default True
        Specify dtype when calling _from_sequence

    Returns
    -------
    result : array-like
        result maybe casted to the dtype.
    """
    ...

def maybe_cast_to_extension_array(
    cls: type[ExtensionArray], obj: ArrayLike, dtype: ExtensionDtype | None = ...
) -> ArrayLike:
    """
    Call to `_from_sequence` that returns the object unchanged on Exception.

    Parameters
    ----------
    cls : class, subclass of ExtensionArray
    obj : arraylike
        Values to pass to cls._from_sequence
    dtype : ExtensionDtype, optional

    Returns
    -------
    ExtensionArray or obj
    """
    ...

@overload
def ensure_dtype_can_hold_na(dtype: np.dtype) -> np.dtype: ...
@overload
def ensure_dtype_can_hold_na(dtype: ExtensionDtype) -> ExtensionDtype: ...
def ensure_dtype_can_hold_na(dtype: DtypeObj) -> DtypeObj:
    """
    If we have a dtype that cannot hold NA values, find the best match that can.
    """
    ...

def maybe_promote(
    dtype: np.dtype, fill_value=...
):  # -> tuple[dtype, Unknown] | tuple[dtype, object | Unknown | float | Literal[0, False]] | tuple[dtype, float | Unknown] | tuple[dtype, Any] | tuple[dtype, datetime64] | tuple[dtype, datetime64 | () -> datetime64] | tuple[dtype, date | str | Unknown | datetime]:
    """
    Find the minimal dtype that can hold both the given dtype and fill_value.

    Parameters
    ----------
    dtype : np.dtype
    fill_value : scalar, default np.nan

    Returns
    -------
    dtype
        Upcasted from dtype argument if necessary.
    fill_value
        Upcasted from fill_value argument if necessary.

    Raises
    ------
    ValueError
        If fill_value is a non-scalar and dtype is not object.
    """
    ...

def infer_dtype_from(val, pandas_dtype: bool = ...) -> tuple[DtypeObj, Any]:
    """
    Interpret the dtype from a scalar or array.

    Parameters
    ----------
    val : object
    pandas_dtype : bool, default False
        whether to infer dtype including pandas extension types.
        If False, scalar/array belongs to pandas extension types is inferred as
        object
    """
    ...

def infer_dtype_from_scalar(val, pandas_dtype: bool = ...) -> tuple[DtypeObj, Any]:
    """
    Interpret the dtype from a scalar.

    Parameters
    ----------
    pandas_dtype : bool, default False
        whether to infer dtype including pandas extension types.
        If False, scalar belongs to pandas extension types is inferred as
        object
    """
    ...

def dict_compat(d: dict[Scalar, Scalar]) -> dict[Scalar, Scalar]:
    """
    Convert datetimelike-keyed dicts to a Timestamp-keyed dict.

    Parameters
    ----------
    d: dict-like object

    Returns
    -------
    dict
    """
    ...

def infer_dtype_from_array(arr, pandas_dtype: bool = ...) -> tuple[DtypeObj, ArrayLike]:
    """
    Infer the dtype from an array.

    Parameters
    ----------
    arr : array
    pandas_dtype : bool, default False
        whether to infer dtype including pandas extension types.
        If False, array belongs to pandas extension types
        is inferred as object

    Returns
    -------
    tuple (numpy-compat/pandas-compat dtype, array)

    Notes
    -----
    if pandas_dtype=False. these infer to numpy dtypes
    exactly with the exception that mixed / object dtypes
    are not coerced by stringifying or conversion

    if pandas_dtype=True. datetime64tz-aware/categorical
    types will retain there character.

    Examples
    --------
    >>> np.asarray([1, '1'])
    array(['1', '1'], dtype='<U21')

    >>> infer_dtype_from_array([1, '1'])
    (dtype('O'), [1, '1'])
    """
    ...

def maybe_infer_dtype_type(element):  # -> None:
    """
    Try to infer an object's dtype, for use in arithmetic ops.

    Uses `element.dtype` if that's available.
    Objects implementing the iterator protocol are cast to a NumPy array,
    and from there the array's type is used.

    Parameters
    ----------
    element : object
        Possibly has a `.dtype` attribute, and possibly the iterator
        protocol.

    Returns
    -------
    tipo : type

    Examples
    --------
    >>> from collections import namedtuple
    >>> Foo = namedtuple("Foo", "dtype")
    >>> maybe_infer_dtype_type(Foo(np.dtype("i8")))
    dtype('int64')
    """
    ...

def maybe_upcast(
    values: NumpyArrayT, fill_value: Scalar = ..., copy: bool = ...
) -> tuple[NumpyArrayT, Scalar]:
    """
    Provide explicit type promotion and coercion.

    Parameters
    ----------
    values : np.ndarray
        The array that we may want to upcast.
    fill_value : what we want to fill with
    copy : bool, default True
        If True always make a copy even if no upcast is required.

    Returns
    -------
    values: np.ndarray
        the original array, possibly upcast
    fill_value:
        the fill value, possibly upcast
    """
    ...

def invalidate_string_dtypes(dtype_set: set[DtypeObj]):  # -> None:
    """
    Change string like dtypes to object for
    ``DataFrame.select_dtypes()``.
    """
    ...

def coerce_indexer_dtype(indexer, categories):  # -> ndarray:
    """coerce the indexer input array to the smallest dtype possible"""
    ...

def astype_dt64_to_dt64tz(
    values: ArrayLike, dtype: DtypeObj, copy: bool, via_utc: bool = ...
) -> DatetimeArray: ...
def astype_td64_unit_conversion(
    values: np.ndarray, dtype: np.dtype, copy: bool
) -> np.ndarray:
    """
    By pandas convention, converting to non-nano timedelta64
    returns an int64-dtyped array with ints representing multiples
    of the desired timedelta unit.  This is essentially division.

    Parameters
    ----------
    values : np.ndarray[timedelta64[ns]]
    dtype : np.dtype
        timedelta64 with unit not-necessarily nano
    copy : bool

    Returns
    -------
    np.ndarray
    """
    ...

@overload
def astype_nansafe(
    arr: np.ndarray, dtype: np.dtype, copy: bool = ..., skipna: bool = ...
) -> np.ndarray: ...
@overload
def astype_nansafe(
    arr: np.ndarray, dtype: ExtensionDtype, copy: bool = ..., skipna: bool = ...
) -> ExtensionArray: ...
def astype_nansafe(
    arr: np.ndarray, dtype: DtypeObj, copy: bool = ..., skipna: bool = ...
) -> ArrayLike:
    """
    Cast the elements of an array to a given dtype a nan-safe manner.

    Parameters
    ----------
    arr : ndarray
    dtype : np.dtype or ExtensionDtype
    copy : bool, default True
        If False, a view will be attempted but may fail, if
        e.g. the item sizes don't align.
    skipna: bool, default False
        Whether or not we should skip NaN when casting as a string-type.

    Raises
    ------
    ValueError
        The dtype was a datetime64/timedelta64 dtype, but it had no unit.
    """
    ...

def astype_float_to_int_nansafe(
    values: np.ndarray, dtype: np.dtype, copy: bool
) -> np.ndarray:
    """
    astype with a check preventing converting NaN to an meaningless integer value.
    """
    ...

def astype_array(values: ArrayLike, dtype: DtypeObj, copy: bool = ...) -> ArrayLike:
    """
    Cast array (ndarray or ExtensionArray) to the new dtype.

    Parameters
    ----------
    values : ndarray or ExtensionArray
    dtype : dtype object
    copy : bool, default False
        copy if indicated

    Returns
    -------
    ndarray or ExtensionArray
    """
    ...

def astype_array_safe(
    values: ArrayLike, dtype, copy: bool = ..., errors: str = ...
) -> ArrayLike:
    """
    Cast array (ndarray or ExtensionArray) to the new dtype.

    This basically is the implementation for DataFrame/Series.astype and
    includes all custom logic for pandas (NaN-safety, converting str to object,
    not allowing )

    Parameters
    ----------
    values : ndarray or ExtensionArray
    dtype : str, dtype convertible
    copy : bool, default False
        copy if indicated
    errors : str, {'raise', 'ignore'}, default 'raise'
        - ``raise`` : allow exceptions to be raised
        - ``ignore`` : suppress exceptions. On error return original object

    Returns
    -------
    ndarray or ExtensionArray
    """
    ...

def soft_convert_objects(
    values: np.ndarray,
    datetime: bool = ...,
    numeric: bool = ...,
    timedelta: bool = ...,
    period: bool = ...,
    copy: bool = ...,
) -> ArrayLike:
    """
    Try to coerce datetime, timedelta, and numeric object-dtype columns
    to inferred dtype.

    Parameters
    ----------
    values : np.ndarray[object]
    datetime : bool, default True
    numeric: bool, default True
    timedelta : bool, default True
    period : bool, default True
    copy : bool, default True

    Returns
    -------
    np.ndarray or ExtensionArray
    """
    ...

def convert_dtypes(
    input_array: ArrayLike,
    convert_string: bool = ...,
    convert_integer: bool = ...,
    convert_boolean: bool = ...,
    convert_floating: bool = ...,
) -> DtypeObj:
    """
    Convert objects to best possible type, and optionally,
    to types supporting ``pd.NA``.

    Parameters
    ----------
    input_array : ExtensionArray or np.ndarray
    convert_string : bool, default True
        Whether object dtypes should be converted to ``StringDtype()``.
    convert_integer : bool, default True
        Whether, if possible, conversion can be done to integer extension types.
    convert_boolean : bool, defaults True
        Whether object dtypes should be converted to ``BooleanDtypes()``.
    convert_floating : bool, defaults True
        Whether, if possible, conversion can be done to floating extension types.
        If `convert_integer` is also True, preference will be give to integer
        dtypes if the floats can be faithfully casted to integers.

    Returns
    -------
    np.dtype, or ExtensionDtype
    """
    ...

def maybe_infer_to_datetimelike(
    value: np.ndarray,
) -> np.ndarray | DatetimeArray | TimedeltaArray | PeriodArray | IntervalArray:
    """
    we might have a array (or single object) that is datetime like,
    and no dtype is passed don't change the value unless we find a
    datetime/timedelta set

    this is pretty strict in that a datetime/timedelta is REQUIRED
    in addition to possible nulls/string likes

    Parameters
    ----------
    value : np.ndarray[object]

    Returns
    -------
    np.ndarray, DatetimeArray, TimedeltaArray, PeriodArray, or IntervalArray

    """
    ...

def maybe_cast_to_datetime(
    value: ExtensionArray | np.ndarray | list, dtype: DtypeObj | None
) -> ExtensionArray | np.ndarray:
    """
    try to cast the array/value to a datetimelike dtype, converting float
    nan to iNaT

    We allow a list *only* when dtype is not None.
    """
    ...

def sanitize_to_nanoseconds(values: np.ndarray, copy: bool = ...) -> np.ndarray:
    """
    Safely convert non-nanosecond datetime64 or timedelta64 values to nanosecond.
    """
    ...

def ensure_nanosecond_dtype(dtype: DtypeObj) -> DtypeObj:
    """
    Convert dtypes with granularity less than nanosecond to nanosecond

    >>> ensure_nanosecond_dtype(np.dtype("M8[s]"))
    dtype('<M8[ns]')

    >>> ensure_nanosecond_dtype(np.dtype("m8[ps]"))
    Traceback (most recent call last):
        ...
    TypeError: cannot convert timedeltalike to dtype [timedelta64[ps]]
    """
    ...

def find_common_type(types: list[DtypeObj]) -> DtypeObj:
    """
    Find a common data type among the given dtypes.

    Parameters
    ----------
    types : list of dtypes

    Returns
    -------
    pandas extension or numpy dtype

    See Also
    --------
    numpy.find_common_type

    """
    ...

def construct_2d_arraylike_from_scalar(
    value: Scalar, length: int, width: int, dtype: np.dtype, copy: bool
) -> np.ndarray: ...
def construct_1d_arraylike_from_scalar(
    value: Scalar, length: int, dtype: DtypeObj | None
) -> ArrayLike:
    """
    create a np.ndarray / pandas type of specified shape and dtype
    filled with values

    Parameters
    ----------
    value : scalar value
    length : int
    dtype : pandas_dtype or np.dtype

    Returns
    -------
    np.ndarray / pandas type of length, filled with value

    """
    ...

def maybe_unbox_datetimelike_tz_deprecation(
    value: Scalar, dtype: DtypeObj, stacklevel: int = ...
):  # -> str | int | float | bool | Period | Timestamp | Timedelta:
    """
    Wrap maybe_unbox_datetimelike with a check for a timezone-aware Timestamp
    along with a timezone-naive datetime64 dtype, which is deprecated.
    """
    ...

def construct_1d_object_array_from_listlike(values: Sized) -> np.ndarray:
    """
    Transform any list-like object in a 1-dimensional numpy array of object
    dtype.

    Parameters
    ----------
    values : any iterable which has a len()

    Raises
    ------
    TypeError
        * If `values` does not have a len()

    Returns
    -------
    1-dimensional numpy array of dtype object
    """
    ...

def maybe_cast_to_integer_array(
    arr: list | np.ndarray, dtype: np.dtype, copy: bool = ...
) -> np.ndarray:
    """
    Takes any dtype and returns the casted version, raising for when data is
    incompatible with integer/unsigned integer dtypes.

    Parameters
    ----------
    arr : np.ndarray or list
        The array to cast.
    dtype : np.dtype
        The integer dtype to cast the array to.
    copy: bool, default False
        Whether to make a copy of the array before returning.

    Returns
    -------
    ndarray
        Array of integer or unsigned integer dtype.

    Raises
    ------
    OverflowError : the dtype is incompatible with the data
    ValueError : loss of precision has occurred during casting

    Examples
    --------
    If you try to coerce negative values to unsigned integers, it raises:

    >>> pd.Series([-1], dtype="uint64")
    Traceback (most recent call last):
        ...
    OverflowError: Trying to coerce negative values to unsigned integers

    Also, if you try to coerce float values to integers, it raises:

    >>> pd.Series([1, 2, 3.5], dtype="int64")
    Traceback (most recent call last):
        ...
    ValueError: Trying to coerce float values to integers
    """
    ...

def convert_scalar_for_putitemlike(scalar: Scalar, dtype: np.dtype) -> Scalar:
    """
    Convert datetimelike scalar if we are setting into a datetime64
    or timedelta64 ndarray.

    Parameters
    ----------
    scalar : scalar
    dtype : np.dtype

    Returns
    -------
    scalar
    """
    ...

def validate_numeric_casting(dtype: np.dtype, value: Scalar) -> None:
    """
    Check that we can losslessly insert the given value into an array
    with the given dtype.

    Parameters
    ----------
    dtype : np.dtype
    value : scalar

    Raises
    ------
    ValueError
    """
    ...

def can_hold_element(arr: ArrayLike, element: Any) -> bool:
    """
    Can we do an inplace setitem with this element in an array with this dtype?

    Parameters
    ----------
    arr : np.ndarray or ExtensionArray
    element : Any

    Returns
    -------
    bool
    """
    ...
