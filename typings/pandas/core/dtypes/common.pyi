from typing import Any, Callable

import numpy as np
from pandas._typing import ArrayLike, DtypeObj, Optional

"""
Common type operations.
"""
DT64NS_DTYPE = ...
TD64NS_DTYPE = ...
INT64_DTYPE = ...
_is_scipy_sparse = ...
ensure_float64 = ...
ensure_float32 = ...

def ensure_float(arr):
    """
    Ensure that an array object has a float dtype if possible.

    Parameters
    ----------
    arr : array-like
        The array whose data type we want to enforce as float.

    Returns
    -------
    float_arr : The original array cast to the float dtype if
                possible. Otherwise, the original array is returned.
    """
    ...

ensure_uint64 = ...
ensure_int64 = ...
ensure_int32 = ...
ensure_int16 = ...
ensure_int8 = ...
ensure_platform_int = ...
ensure_object = ...

def ensure_str(value: bytes | Any) -> str:
    """
    Ensure that bytes and non-strings get converted into ``str`` objects.
    """
    ...

def ensure_python_int(value: int | np.integer) -> int:
    """
    Ensure that a value is a python int.

    Parameters
    ----------
    value: int or numpy.integer

    Returns
    -------
    int

    Raises
    ------
    TypeError: if the value isn't an int or can't be converted to one.
    """
    ...

def classes(*klasses) -> Callable:
    """evaluate if the tipo is a subclass of the klasses"""
    ...

def classes_and_not_datetimelike(*klasses) -> Callable:
    """
    evaluate if the tipo is a subclass of the klasses
    and not a datetimelike
    """
    ...

def is_object_dtype(arr_or_dtype) -> bool:
    """
    Check whether an array-like or dtype is of the object dtype.

    Parameters
    ----------
    arr_or_dtype : array-like
        The array-like or dtype to check.

    Returns
    -------
    boolean
        Whether or not the array-like or dtype is of the object dtype.

    Examples
    --------
    >>> is_object_dtype(object)
    True
    >>> is_object_dtype(int)
    False
    >>> is_object_dtype(np.array([], dtype=object))
    True
    >>> is_object_dtype(np.array([], dtype=int))
    False
    >>> is_object_dtype([1, 2, 3])
    False
    """
    ...

def is_sparse(arr) -> bool:
    """
    Check whether an array-like is a 1-D pandas sparse array.

    Check that the one-dimensional array-like is a pandas sparse array.
    Returns True if it is a pandas sparse array, not another type of
    sparse array.

    Parameters
    ----------
    arr : array-like
        Array-like to check.

    Returns
    -------
    bool
        Whether or not the array-like is a pandas sparse array.

    Examples
    --------
    Returns `True` if the parameter is a 1-D pandas sparse array.

    >>> is_sparse(pd.arrays.SparseArray([0, 0, 1, 0]))
    True
    >>> is_sparse(pd.Series(pd.arrays.SparseArray([0, 0, 1, 0])))
    True

    Returns `False` if the parameter is not sparse.

    >>> is_sparse(np.array([0, 0, 1, 0]))
    False
    >>> is_sparse(pd.Series([0, 1, 0, 0]))
    False

    Returns `False` if the parameter is not a pandas sparse array.

    >>> from scipy.sparse import bsr_matrix
    >>> is_sparse(bsr_matrix([0, 1, 0, 0]))
    False

    Returns `False` if the parameter has more than one dimension.
    """
    ...

def is_scipy_sparse(arr) -> bool:
    """
    Check whether an array-like is a scipy.sparse.spmatrix instance.

    Parameters
    ----------
    arr : array-like
        The array-like to check.

    Returns
    -------
    boolean
        Whether or not the array-like is a scipy.sparse.spmatrix instance.

    Notes
    -----
    If scipy is not installed, this function will always return False.

    Examples
    --------
    >>> from scipy.sparse import bsr_matrix
    >>> is_scipy_sparse(bsr_matrix([1, 2, 3]))
    True
    >>> is_scipy_sparse(pd.arrays.SparseArray([1, 2, 3]))
    False
    """
    ...

def is_categorical(arr) -> bool:
    """
    Check whether an array-like is a Categorical instance.

    Parameters
    ----------
    arr : array-like
        The array-like to check.

    Returns
    -------
    boolean
        Whether or not the array-like is of a Categorical instance.

    Examples
    --------
    >>> is_categorical([1, 2, 3])
    False

    Categoricals, Series Categoricals, and CategoricalIndex will return True.

    >>> cat = pd.Categorical([1, 2, 3])
    >>> is_categorical(cat)
    True
    >>> is_categorical(pd.Series(cat))
    True
    >>> is_categorical(pd.CategoricalIndex([1, 2, 3]))
    True
    """
    ...

def is_datetime64_dtype(arr_or_dtype) -> bool:
    """
    Check whether an array-like or dtype is of the datetime64 dtype.

    Parameters
    ----------
    arr_or_dtype : array-like
        The array-like or dtype to check.

    Returns
    -------
    boolean
        Whether or not the array-like or dtype is of the datetime64 dtype.

    Examples
    --------
    >>> is_datetime64_dtype(object)
    False
    >>> is_datetime64_dtype(np.datetime64)
    True
    >>> is_datetime64_dtype(np.array([], dtype=int))
    False
    >>> is_datetime64_dtype(np.array([], dtype=np.datetime64))
    True
    >>> is_datetime64_dtype([1, 2, 3])
    False
    """
    ...

def is_datetime64tz_dtype(arr_or_dtype) -> bool:
    """
    Check whether an array-like or dtype is of a DatetimeTZDtype dtype.

    Parameters
    ----------
    arr_or_dtype : array-like
        The array-like or dtype to check.

    Returns
    -------
    boolean
        Whether or not the array-like or dtype is of a DatetimeTZDtype dtype.

    Examples
    --------
    >>> is_datetime64tz_dtype(object)
    False
    >>> is_datetime64tz_dtype([1, 2, 3])
    False
    >>> is_datetime64tz_dtype(pd.DatetimeIndex([1, 2, 3]))  # tz-naive
    False
    >>> is_datetime64tz_dtype(pd.DatetimeIndex([1, 2, 3], tz="US/Eastern"))
    True

    >>> dtype = DatetimeTZDtype("ns", tz="US/Eastern")
    >>> s = pd.Series([], dtype=dtype)
    >>> is_datetime64tz_dtype(dtype)
    True
    >>> is_datetime64tz_dtype(s)
    True
    """
    ...

def is_timedelta64_dtype(arr_or_dtype) -> bool:
    """
    Check whether an array-like or dtype is of the timedelta64 dtype.

    Parameters
    ----------
    arr_or_dtype : array-like
        The array-like or dtype to check.

    Returns
    -------
    boolean
        Whether or not the array-like or dtype is of the timedelta64 dtype.

    Examples
    --------
    >>> is_timedelta64_dtype(object)
    False
    >>> is_timedelta64_dtype(np.timedelta64)
    True
    >>> is_timedelta64_dtype([1, 2, 3])
    False
    >>> is_timedelta64_dtype(pd.Series([], dtype="timedelta64[ns]"))
    True
    >>> is_timedelta64_dtype('0 days')
    False
    """
    ...

def is_period_dtype(arr_or_dtype) -> bool:
    """
    Check whether an array-like or dtype is of the Period dtype.

    Parameters
    ----------
    arr_or_dtype : array-like
        The array-like or dtype to check.

    Returns
    -------
    boolean
        Whether or not the array-like or dtype is of the Period dtype.

    Examples
    --------
    >>> is_period_dtype(object)
    False
    >>> is_period_dtype(PeriodDtype(freq="D"))
    True
    >>> is_period_dtype([1, 2, 3])
    False
    >>> is_period_dtype(pd.Period("2017-01-01"))
    False
    >>> is_period_dtype(pd.PeriodIndex([], freq="A"))
    True
    """
    ...

def is_interval_dtype(arr_or_dtype) -> bool:
    """
    Check whether an array-like or dtype is of the Interval dtype.

    Parameters
    ----------
    arr_or_dtype : array-like
        The array-like or dtype to check.

    Returns
    -------
    boolean
        Whether or not the array-like or dtype is of the Interval dtype.

    Examples
    --------
    >>> is_interval_dtype(object)
    False
    >>> is_interval_dtype(IntervalDtype())
    True
    >>> is_interval_dtype([1, 2, 3])
    False
    >>>
    >>> interval = pd.Interval(1, 2, closed="right")
    >>> is_interval_dtype(interval)
    False
    >>> is_interval_dtype(pd.IntervalIndex([interval]))
    True
    """
    ...

def is_categorical_dtype(arr_or_dtype) -> bool:
    """
    Check whether an array-like or dtype is of the Categorical dtype.

    Parameters
    ----------
    arr_or_dtype : array-like
        The array-like or dtype to check.

    Returns
    -------
    boolean
        Whether or not the array-like or dtype is of the Categorical dtype.

    Examples
    --------
    >>> is_categorical_dtype(object)
    False
    >>> is_categorical_dtype(CategoricalDtype())
    True
    >>> is_categorical_dtype([1, 2, 3])
    False
    >>> is_categorical_dtype(pd.Categorical([1, 2, 3]))
    True
    >>> is_categorical_dtype(pd.CategoricalIndex([1, 2, 3]))
    True
    """
    ...

def is_string_dtype(arr_or_dtype) -> bool:
    """
    Check whether the provided array or dtype is of the string dtype.

    Parameters
    ----------
    arr_or_dtype : array-like
        The array or dtype to check.

    Returns
    -------
    boolean
        Whether or not the array or dtype is of the string dtype.

    Examples
    --------
    >>> is_string_dtype(str)
    True
    >>> is_string_dtype(object)
    True
    >>> is_string_dtype(int)
    False
    >>>
    >>> is_string_dtype(np.array(['a', 'b']))
    True
    >>> is_string_dtype(pd.Series([1, 2]))
    False
    """
    ...

def is_dtype_equal(source, target) -> bool:
    """
    Check if two dtypes are equal.

    Parameters
    ----------
    source : The first dtype to compare
    target : The second dtype to compare

    Returns
    -------
    boolean
        Whether or not the two dtypes are equal.

    Examples
    --------
    >>> is_dtype_equal(int, float)
    False
    >>> is_dtype_equal("int", int)
    True
    >>> is_dtype_equal(object, "category")
    False
    >>> is_dtype_equal(CategoricalDtype(), "category")
    True
    >>> is_dtype_equal(DatetimeTZDtype(tz="UTC"), "datetime64")
    False
    """
    ...

def is_any_int_dtype(arr_or_dtype) -> bool:
    """
    Check whether the provided array or dtype is of an integer dtype.

    In this function, timedelta64 instances are also considered "any-integer"
    type objects and will return True.

    This function is internal and should not be exposed in the public API.

    The nullable Integer dtypes (e.g. pandas.Int64Dtype) are also considered
    as integer by this function.

    Parameters
    ----------
    arr_or_dtype : array-like
        The array or dtype to check.

    Returns
    -------
    boolean
        Whether or not the array or dtype is of an integer dtype.

    Examples
    --------
    >>> is_any_int_dtype(str)
    False
    >>> is_any_int_dtype(int)
    True
    >>> is_any_int_dtype(float)
    False
    >>> is_any_int_dtype(np.uint64)
    True
    >>> is_any_int_dtype(np.datetime64)
    False
    >>> is_any_int_dtype(np.timedelta64)
    True
    >>> is_any_int_dtype(np.array(['a', 'b']))
    False
    >>> is_any_int_dtype(pd.Series([1, 2]))
    True
    >>> is_any_int_dtype(np.array([], dtype=np.timedelta64))
    True
    >>> is_any_int_dtype(pd.Index([1, 2.]))  # float
    False
    """
    ...

def is_integer_dtype(arr_or_dtype) -> bool:
    """
    Check whether the provided array or dtype is of an integer dtype.

    Unlike in `in_any_int_dtype`, timedelta64 instances will return False.

    The nullable Integer dtypes (e.g. pandas.Int64Dtype) are also considered
    as integer by this function.

    Parameters
    ----------
    arr_or_dtype : array-like
        The array or dtype to check.

    Returns
    -------
    boolean
        Whether or not the array or dtype is of an integer dtype and
        not an instance of timedelta64.

    Examples
    --------
    >>> is_integer_dtype(str)
    False
    >>> is_integer_dtype(int)
    True
    >>> is_integer_dtype(float)
    False
    >>> is_integer_dtype(np.uint64)
    True
    >>> is_integer_dtype('int8')
    True
    >>> is_integer_dtype('Int8')
    True
    >>> is_integer_dtype(pd.Int8Dtype)
    True
    >>> is_integer_dtype(np.datetime64)
    False
    >>> is_integer_dtype(np.timedelta64)
    False
    >>> is_integer_dtype(np.array(['a', 'b']))
    False
    >>> is_integer_dtype(pd.Series([1, 2]))
    True
    >>> is_integer_dtype(np.array([], dtype=np.timedelta64))
    False
    >>> is_integer_dtype(pd.Index([1, 2.]))  # float
    False
    """
    ...

def is_signed_integer_dtype(arr_or_dtype) -> bool:
    """
    Check whether the provided array or dtype is of a signed integer dtype.

    Unlike in `in_any_int_dtype`, timedelta64 instances will return False.

    The nullable Integer dtypes (e.g. pandas.Int64Dtype) are also considered
    as integer by this function.

    Parameters
    ----------
    arr_or_dtype : array-like
        The array or dtype to check.

    Returns
    -------
    boolean
        Whether or not the array or dtype is of a signed integer dtype
        and not an instance of timedelta64.

    Examples
    --------
    >>> is_signed_integer_dtype(str)
    False
    >>> is_signed_integer_dtype(int)
    True
    >>> is_signed_integer_dtype(float)
    False
    >>> is_signed_integer_dtype(np.uint64)  # unsigned
    False
    >>> is_signed_integer_dtype('int8')
    True
    >>> is_signed_integer_dtype('Int8')
    True
    >>> is_signed_integer_dtype(pd.Int8Dtype)
    True
    >>> is_signed_integer_dtype(np.datetime64)
    False
    >>> is_signed_integer_dtype(np.timedelta64)
    False
    >>> is_signed_integer_dtype(np.array(['a', 'b']))
    False
    >>> is_signed_integer_dtype(pd.Series([1, 2]))
    True
    >>> is_signed_integer_dtype(np.array([], dtype=np.timedelta64))
    False
    >>> is_signed_integer_dtype(pd.Index([1, 2.]))  # float
    False
    >>> is_signed_integer_dtype(np.array([1, 2], dtype=np.uint32))  # unsigned
    False
    """
    ...

def is_unsigned_integer_dtype(arr_or_dtype) -> bool:
    """
    Check whether the provided array or dtype is of an unsigned integer dtype.

    The nullable Integer dtypes (e.g. pandas.UInt64Dtype) are also
    considered as integer by this function.

    Parameters
    ----------
    arr_or_dtype : array-like
        The array or dtype to check.

    Returns
    -------
    boolean
        Whether or not the array or dtype is of an unsigned integer dtype.

    Examples
    --------
    >>> is_unsigned_integer_dtype(str)
    False
    >>> is_unsigned_integer_dtype(int)  # signed
    False
    >>> is_unsigned_integer_dtype(float)
    False
    >>> is_unsigned_integer_dtype(np.uint64)
    True
    >>> is_unsigned_integer_dtype('uint8')
    True
    >>> is_unsigned_integer_dtype('UInt8')
    True
    >>> is_unsigned_integer_dtype(pd.UInt8Dtype)
    True
    >>> is_unsigned_integer_dtype(np.array(['a', 'b']))
    False
    >>> is_unsigned_integer_dtype(pd.Series([1, 2]))  # signed
    False
    >>> is_unsigned_integer_dtype(pd.Index([1, 2.]))  # float
    False
    >>> is_unsigned_integer_dtype(np.array([1, 2], dtype=np.uint32))
    True
    """
    ...

def is_int64_dtype(arr_or_dtype) -> bool:
    """
    Check whether the provided array or dtype is of the int64 dtype.

    Parameters
    ----------
    arr_or_dtype : array-like
        The array or dtype to check.

    Returns
    -------
    boolean
        Whether or not the array or dtype is of the int64 dtype.

    Notes
    -----
    Depending on system architecture, the return value of `is_int64_dtype(
    int)` will be True if the OS uses 64-bit integers and False if the OS
    uses 32-bit integers.

    Examples
    --------
    >>> is_int64_dtype(str)
    False
    >>> is_int64_dtype(np.int32)
    False
    >>> is_int64_dtype(np.int64)
    True
    >>> is_int64_dtype('int8')
    False
    >>> is_int64_dtype('Int8')
    False
    >>> is_int64_dtype(pd.Int64Dtype)
    True
    >>> is_int64_dtype(float)
    False
    >>> is_int64_dtype(np.uint64)  # unsigned
    False
    >>> is_int64_dtype(np.array(['a', 'b']))
    False
    >>> is_int64_dtype(np.array([1, 2], dtype=np.int64))
    True
    >>> is_int64_dtype(pd.Index([1, 2.]))  # float
    False
    >>> is_int64_dtype(np.array([1, 2], dtype=np.uint32))  # unsigned
    False
    """
    ...

def is_datetime64_any_dtype(arr_or_dtype) -> bool:
    """
    Check whether the provided array or dtype is of the datetime64 dtype.

    Parameters
    ----------
    arr_or_dtype : array-like
        The array or dtype to check.

    Returns
    -------
    bool
        Whether or not the array or dtype is of the datetime64 dtype.

    Examples
    --------
    >>> is_datetime64_any_dtype(str)
    False
    >>> is_datetime64_any_dtype(int)
    False
    >>> is_datetime64_any_dtype(np.datetime64)  # can be tz-naive
    True
    >>> is_datetime64_any_dtype(DatetimeTZDtype("ns", "US/Eastern"))
    True
    >>> is_datetime64_any_dtype(np.array(['a', 'b']))
    False
    >>> is_datetime64_any_dtype(np.array([1, 2]))
    False
    >>> is_datetime64_any_dtype(np.array([], dtype="datetime64[ns]"))
    True
    >>> is_datetime64_any_dtype(pd.DatetimeIndex([1, 2, 3], dtype="datetime64[ns]"))
    True
    """
    ...

def is_datetime64_ns_dtype(arr_or_dtype) -> bool:
    """
    Check whether the provided array or dtype is of the datetime64[ns] dtype.

    Parameters
    ----------
    arr_or_dtype : array-like
        The array or dtype to check.

    Returns
    -------
    bool
        Whether or not the array or dtype is of the datetime64[ns] dtype.

    Examples
    --------
    >>> is_datetime64_ns_dtype(str)
    False
    >>> is_datetime64_ns_dtype(int)
    False
    >>> is_datetime64_ns_dtype(np.datetime64)  # no unit
    False
    >>> is_datetime64_ns_dtype(DatetimeTZDtype("ns", "US/Eastern"))
    True
    >>> is_datetime64_ns_dtype(np.array(['a', 'b']))
    False
    >>> is_datetime64_ns_dtype(np.array([1, 2]))
    False
    >>> is_datetime64_ns_dtype(np.array([], dtype="datetime64"))  # no unit
    False
    >>> is_datetime64_ns_dtype(np.array([], dtype="datetime64[ps]"))  # wrong unit
    False
    >>> is_datetime64_ns_dtype(pd.DatetimeIndex([1, 2, 3], dtype="datetime64[ns]"))
    True
    """
    ...

def is_timedelta64_ns_dtype(arr_or_dtype) -> bool:
    """
    Check whether the provided array or dtype is of the timedelta64[ns] dtype.

    This is a very specific dtype, so generic ones like `np.timedelta64`
    will return False if passed into this function.

    Parameters
    ----------
    arr_or_dtype : array-like
        The array or dtype to check.

    Returns
    -------
    boolean
        Whether or not the array or dtype is of the timedelta64[ns] dtype.

    Examples
    --------
    >>> is_timedelta64_ns_dtype(np.dtype('m8[ns]'))
    True
    >>> is_timedelta64_ns_dtype(np.dtype('m8[ps]'))  # Wrong frequency
    False
    >>> is_timedelta64_ns_dtype(np.array([1, 2], dtype='m8[ns]'))
    True
    >>> is_timedelta64_ns_dtype(np.array([1, 2], dtype=np.timedelta64))
    False
    """
    ...

def is_datetime_or_timedelta_dtype(arr_or_dtype) -> bool:
    """
    Check whether the provided array or dtype is of
    a timedelta64 or datetime64 dtype.

    Parameters
    ----------
    arr_or_dtype : array-like
        The array or dtype to check.

    Returns
    -------
    boolean
        Whether or not the array or dtype is of a timedelta64,
        or datetime64 dtype.

    Examples
    --------
    >>> is_datetime_or_timedelta_dtype(str)
    False
    >>> is_datetime_or_timedelta_dtype(int)
    False
    >>> is_datetime_or_timedelta_dtype(np.datetime64)
    True
    >>> is_datetime_or_timedelta_dtype(np.timedelta64)
    True
    >>> is_datetime_or_timedelta_dtype(np.array(['a', 'b']))
    False
    >>> is_datetime_or_timedelta_dtype(pd.Series([1, 2]))
    False
    >>> is_datetime_or_timedelta_dtype(np.array([], dtype=np.timedelta64))
    True
    >>> is_datetime_or_timedelta_dtype(np.array([], dtype=np.datetime64))
    True
    """
    ...

def is_numeric_v_string_like(a: ArrayLike, b):  # -> bool:
    """
    Check if we are comparing a string-like object to a numeric ndarray.
    NumPy doesn't like to compare such objects, especially numeric arrays
    and scalar string-likes.

    Parameters
    ----------
    a : array-like
        The first object to check.
    b : array-like, scalar
        The second object to check.

    Returns
    -------
    boolean
        Whether we return a comparing a string-like object to a numeric array.

    Examples
    --------
    >>> is_numeric_v_string_like(np.array([1]), "foo")
    True
    >>> is_numeric_v_string_like(np.array([1, 2]), np.array(["foo"]))
    True
    >>> is_numeric_v_string_like(np.array(["foo"]), np.array([1, 2]))
    True
    >>> is_numeric_v_string_like(np.array([1]), np.array([2]))
    False
    >>> is_numeric_v_string_like(np.array(["foo"]), np.array(["foo"]))
    False
    """
    ...

def is_datetimelike_v_numeric(a, b):  # -> bool:
    """
    Check if we are comparing a datetime-like object to a numeric object.
    By "numeric," we mean an object that is either of an int or float dtype.

    Parameters
    ----------
    a : array-like, scalar
        The first object to check.
    b : array-like, scalar
        The second object to check.

    Returns
    -------
    boolean
        Whether we return a comparing a datetime-like to a numeric object.

    Examples
    --------
    >>> from datetime import datetime
    >>> dt = np.datetime64(datetime(2017, 1, 1))
    >>>
    >>> is_datetimelike_v_numeric(1, 1)
    False
    >>> is_datetimelike_v_numeric(dt, dt)
    False
    >>> is_datetimelike_v_numeric(1, dt)
    True
    >>> is_datetimelike_v_numeric(dt, 1)  # symmetric check
    True
    >>> is_datetimelike_v_numeric(np.array([dt]), 1)
    True
    >>> is_datetimelike_v_numeric(np.array([1]), dt)
    True
    >>> is_datetimelike_v_numeric(np.array([dt]), np.array([1]))
    True
    >>> is_datetimelike_v_numeric(np.array([1]), np.array([2]))
    False
    >>> is_datetimelike_v_numeric(np.array([dt]), np.array([dt]))
    False
    """
    ...

def needs_i8_conversion(arr_or_dtype) -> bool:
    """
    Check whether the array or dtype should be converted to int64.

    An array-like or dtype "needs" such a conversion if the array-like
    or dtype is of a datetime-like dtype

    Parameters
    ----------
    arr_or_dtype : array-like
        The array or dtype to check.

    Returns
    -------
    boolean
        Whether or not the array or dtype should be converted to int64.

    Examples
    --------
    >>> needs_i8_conversion(str)
    False
    >>> needs_i8_conversion(np.int64)
    False
    >>> needs_i8_conversion(np.datetime64)
    True
    >>> needs_i8_conversion(np.array(['a', 'b']))
    False
    >>> needs_i8_conversion(pd.Series([1, 2]))
    False
    >>> needs_i8_conversion(pd.Series([], dtype="timedelta64[ns]"))
    True
    >>> needs_i8_conversion(pd.DatetimeIndex([1, 2, 3], tz="US/Eastern"))
    True
    """
    ...

def is_numeric_dtype(arr_or_dtype) -> bool:
    """
    Check whether the provided array or dtype is of a numeric dtype.

    Parameters
    ----------
    arr_or_dtype : array-like
        The array or dtype to check.

    Returns
    -------
    boolean
        Whether or not the array or dtype is of a numeric dtype.

    Examples
    --------
    >>> is_numeric_dtype(str)
    False
    >>> is_numeric_dtype(int)
    True
    >>> is_numeric_dtype(float)
    True
    >>> is_numeric_dtype(np.uint64)
    True
    >>> is_numeric_dtype(np.datetime64)
    False
    >>> is_numeric_dtype(np.timedelta64)
    False
    >>> is_numeric_dtype(np.array(['a', 'b']))
    False
    >>> is_numeric_dtype(pd.Series([1, 2]))
    True
    >>> is_numeric_dtype(pd.Index([1, 2.]))
    True
    >>> is_numeric_dtype(np.array([], dtype=np.timedelta64))
    False
    """
    ...

def is_float_dtype(arr_or_dtype) -> bool:
    """
    Check whether the provided array or dtype is of a float dtype.

    This function is internal and should not be exposed in the public API.

    Parameters
    ----------
    arr_or_dtype : array-like
        The array or dtype to check.

    Returns
    -------
    boolean
        Whether or not the array or dtype is of a float dtype.

    Examples
    --------
    >>> is_float_dtype(str)
    False
    >>> is_float_dtype(int)
    False
    >>> is_float_dtype(float)
    True
    >>> is_float_dtype(np.array(['a', 'b']))
    False
    >>> is_float_dtype(pd.Series([1, 2]))
    False
    >>> is_float_dtype(pd.Index([1, 2.]))
    True
    """
    ...

def is_bool_dtype(arr_or_dtype) -> bool:
    """
    Check whether the provided array or dtype is of a boolean dtype.

    Parameters
    ----------
    arr_or_dtype : array-like
        The array or dtype to check.

    Returns
    -------
    boolean
        Whether or not the array or dtype is of a boolean dtype.

    Notes
    -----
    An ExtensionArray is considered boolean when the ``_is_boolean``
    attribute is set to True.

    Examples
    --------
    >>> is_bool_dtype(str)
    False
    >>> is_bool_dtype(int)
    False
    >>> is_bool_dtype(bool)
    True
    >>> is_bool_dtype(np.bool_)
    True
    >>> is_bool_dtype(np.array(['a', 'b']))
    False
    >>> is_bool_dtype(pd.Series([1, 2]))
    False
    >>> is_bool_dtype(np.array([True, False]))
    True
    >>> is_bool_dtype(pd.Categorical([True, False]))
    True
    >>> is_bool_dtype(pd.arrays.SparseArray([True, False]))
    True
    """
    ...

def is_extension_type(arr) -> bool:
    """
    Check whether an array-like is of a pandas extension class instance.

    .. deprecated:: 1.0.0
        Use ``is_extension_array_dtype`` instead.

    Extension classes include categoricals, pandas sparse objects (i.e.
    classes represented within the pandas library and not ones external
    to it like scipy sparse matrices), and datetime-like arrays.

    Parameters
    ----------
    arr : array-like
        The array-like to check.

    Returns
    -------
    boolean
        Whether or not the array-like is of a pandas extension class instance.

    Examples
    --------
    >>> is_extension_type([1, 2, 3])
    False
    >>> is_extension_type(np.array([1, 2, 3]))
    False
    >>>
    >>> cat = pd.Categorical([1, 2, 3])
    >>>
    >>> is_extension_type(cat)
    True
    >>> is_extension_type(pd.Series(cat))
    True
    >>> is_extension_type(pd.arrays.SparseArray([1, 2, 3]))
    True
    >>> from scipy.sparse import bsr_matrix
    >>> is_extension_type(bsr_matrix([1, 2, 3]))
    False
    >>> is_extension_type(pd.DatetimeIndex([1, 2, 3]))
    False
    >>> is_extension_type(pd.DatetimeIndex([1, 2, 3], tz="US/Eastern"))
    True
    >>>
    >>> dtype = DatetimeTZDtype("ns", tz="US/Eastern")
    >>> s = pd.Series([], dtype=dtype)
    >>> is_extension_type(s)
    True
    """
    ...

def is_1d_only_ea_obj(obj: Any) -> bool:
    """
    ExtensionArray that does not support 2D, or more specifically that does
    not use HybridBlock.
    """
    ...

def is_1d_only_ea_dtype(dtype: Optional[DtypeObj]) -> bool:
    """
    Analogue to is_extension_array_dtype but excluding DatetimeTZDtype.
    """
    ...

def is_extension_array_dtype(arr_or_dtype) -> bool:
    """
    Check if an object is a pandas extension array type.

    See the :ref:`Use Guide <extending.extension-types>` for more.

    Parameters
    ----------
    arr_or_dtype : object
        For array-like input, the ``.dtype`` attribute will
        be extracted.

    Returns
    -------
    bool
        Whether the `arr_or_dtype` is an extension array type.

    Notes
    -----
    This checks whether an object implements the pandas extension
    array interface. In pandas, this includes:

    * Categorical
    * Sparse
    * Interval
    * Period
    * DatetimeArray
    * TimedeltaArray

    Third-party libraries may implement arrays or types satisfying
    this interface as well.

    Examples
    --------
    >>> from pandas.api.types import is_extension_array_dtype
    >>> arr = pd.Categorical(['a', 'b'])
    >>> is_extension_array_dtype(arr)
    True
    >>> is_extension_array_dtype(arr.dtype)
    True

    >>> arr = np.array(['a', 'b'])
    >>> is_extension_array_dtype(arr.dtype)
    False
    """
    ...

def is_ea_or_datetimelike_dtype(dtype: Optional[DtypeObj]) -> bool:
    """
    Check for ExtensionDtype, datetime64 dtype, or timedelta64 dtype.

    Notes
    -----
    Checks only for dtype objects, not dtype-castable strings or types.
    """
    ...

def is_complex_dtype(arr_or_dtype) -> bool:
    """
    Check whether the provided array or dtype is of a complex dtype.

    Parameters
    ----------
    arr_or_dtype : array-like
        The array or dtype to check.

    Returns
    -------
    boolean
        Whether or not the array or dtype is of a complex dtype.

    Examples
    --------
    >>> is_complex_dtype(str)
    False
    >>> is_complex_dtype(int)
    False
    >>> is_complex_dtype(np.complex_)
    True
    >>> is_complex_dtype(np.array(['a', 'b']))
    False
    >>> is_complex_dtype(pd.Series([1, 2]))
    False
    >>> is_complex_dtype(np.array([1 + 1j, 5]))
    True
    """
    ...

def get_dtype(arr_or_dtype) -> DtypeObj:
    """
    Get the dtype instance associated with an array
    or dtype object.

    Parameters
    ----------
    arr_or_dtype : array-like
        The array-like or dtype object whose dtype we want to extract.

    Returns
    -------
    obj_dtype : The extract dtype instance from the
                passed in array or dtype object.

    Raises
    ------
    TypeError : The passed in object is None.
    """
    ...

def infer_dtype_from_object(dtype) -> DtypeObj:
    """
    Get a numpy dtype.type-style object for a dtype object.

    This methods also includes handling of the datetime64[ns] and
    datetime64[ns, TZ] objects.

    If no dtype can be found, we return ``object``.

    Parameters
    ----------
    dtype : dtype, type
        The dtype object whose numpy dtype.type-style
        object we want to extract.

    Returns
    -------
    dtype_object : The extracted numpy dtype.type-style object.
    """
    ...

def validate_all_hashable(*args, error_name: Optional[str] = ...) -> None:
    """
    Return None if all args are hashable, else raise a TypeError.

    Parameters
    ----------
    *args
        Arguments to validate.
    error_name : str, optional
        The name to use if error

    Raises
    ------
    TypeError : If an argument is not hashable

    Returns
    -------
    None
    """
    ...

def pandas_dtype(dtype) -> DtypeObj:
    """
    Convert input into a pandas only dtype object or a numpy dtype object.

    Parameters
    ----------
    dtype : object to be converted

    Returns
    -------
    np.dtype or a pandas dtype

    Raises
    ------
    TypeError if not a dtype
    """
    ...
