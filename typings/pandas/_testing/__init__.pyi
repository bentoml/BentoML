from __future__ import annotations

import collections
import operator
import os
import re
import string
import warnings
from datetime import datetime
from decimal import Decimal
from functools import wraps
from typing import TYPE_CHECKING, Callable, ContextManager, Counter, Iterable

import numpy as np
import pandas as pd
from pandas import (
    Categorical,
    CategoricalIndex,
    DataFrame,
    DatetimeIndex,
    Float64Index,
    Index,
    Int64Index,
    IntervalIndex,
    MultiIndex,
    PeriodIndex,
    RangeIndex,
    Series,
    TimedeltaIndex,
    UInt64Index,
    bdate_range,
)
from pandas._config.localization import can_set_locale, get_locales, set_locale
from pandas._testing._io import (
    close,
    network,
    round_trip_localpath,
    round_trip_pathlib,
    round_trip_pickle,
    with_connectivity_check,
    write_to_compressed,
)
from pandas._testing._random import randbool, rands, rands_array, randu_array
from pandas._testing._warnings import assert_produces_warning
from pandas._testing.asserters import (
    assert_almost_equal,
    assert_attr_equal,
    assert_categorical_equal,
    assert_class_equal,
    assert_contains_all,
    assert_copy,
    assert_datetime_array_equal,
    assert_dict_equal,
    assert_equal,
    assert_extension_array_equal,
    assert_frame_equal,
    assert_index_equal,
    assert_interval_array_equal,
    assert_is_sorted,
    assert_is_valid_plot_return_object,
    assert_numpy_array_equal,
    assert_period_array_equal,
    assert_series_equal,
    assert_sp_array_equal,
    assert_timedelta_array_equal,
    raise_assert_detail,
)
from pandas._testing.compat import get_dtype
from pandas._testing.contexts import (
    RNGContext,
    decompress_file,
    ensure_clean,
    ensure_clean_dir,
    ensure_safe_environment_variables,
    set_timezone,
    use_numexpr,
    with_csv_dialect,
)
from pandas._typing import Dtype
from pandas.core.arrays import (
    DatetimeArray,
    PandasArray,
    PeriodArray,
    TimedeltaArray,
    period_array,
)
from pandas.core.dtypes.common import (
    is_datetime64_dtype,
    is_datetime64tz_dtype,
    is_float_dtype,
    is_integer_dtype,
    is_period_dtype,
    is_sequence,
    is_timedelta64_dtype,
    is_unsigned_integer_dtype,
    pandas_dtype,
)

if TYPE_CHECKING: ...
_N = ...
_K = ...
UNSIGNED_INT_DTYPES: list[Dtype] = ...
UNSIGNED_EA_INT_DTYPES: list[Dtype] = ...
SIGNED_INT_DTYPES: list[Dtype] = ...
SIGNED_EA_INT_DTYPES: list[Dtype] = ...
ALL_INT_DTYPES = ...
ALL_EA_INT_DTYPES = ...
FLOAT_DTYPES: list[Dtype] = ...
FLOAT_EA_DTYPES: list[Dtype] = ...
COMPLEX_DTYPES: list[Dtype] = ...
STRING_DTYPES: list[Dtype] = ...
DATETIME64_DTYPES: list[Dtype] = ...
TIMEDELTA64_DTYPES: list[Dtype] = ...
BOOL_DTYPES: list[Dtype] = ...
BYTES_DTYPES: list[Dtype] = ...
OBJECT_DTYPES: list[Dtype] = ...
ALL_REAL_DTYPES = ...
ALL_NUMPY_DTYPES = ...
NULL_OBJECTS = ...
EMPTY_STRING_PATTERN = ...
_testing_mode_warnings = ...

def set_testing_mode(): ...
def reset_testing_mode(): ...
def reset_display_options():  # -> None:
    """
    Reset the display options for printing and representing objects.
    """
    ...

def equalContents(arr1, arr2) -> bool:
    """
    Checks if the set of unique elements of arr1 and arr2 are equivalent.
    """
    ...

def box_expected(expected, box_cls, transpose=...):
    """
    Helper function to wrap the expected output of a test in a given box_class.

    Parameters
    ----------
    expected : np.ndarray, Index, Series
    box_cls : {Index, Series, DataFrame}

    Returns
    -------
    subclass of box_cls
    """
    ...

def to_array(obj): ...
def getCols(k): ...
def makeStringIndex(k=..., name=...): ...
def makeUnicodeIndex(k=..., name=...): ...
def makeCategoricalIndex(k=..., n=..., name=..., **kwargs):  # -> CategoricalIndex:
    """make a length k index or n categories"""
    ...

def makeIntervalIndex(k=..., name=..., **kwargs):  # -> IntervalIndex:
    """make a length k IntervalIndex"""
    ...

def makeBoolIndex(k=..., name=...): ...
def makeNumericIndex(k=..., name=..., *, dtype): ...
def makeIntIndex(k=..., name=...): ...
def makeUIntIndex(k=..., name=...): ...
def makeRangeIndex(k=..., name=..., **kwargs): ...
def makeFloatIndex(k=..., name=...): ...
def makeDateIndex(k: int = ..., freq=..., name=..., **kwargs) -> DatetimeIndex: ...
def makeTimedeltaIndex(k: int = ..., freq=..., name=..., **kwargs) -> TimedeltaIndex: ...
def makePeriodIndex(k: int = ..., name=..., **kwargs) -> PeriodIndex: ...
def makeMultiIndex(k=..., names=..., **kwargs): ...

_names = ...

def index_subclass_makers_generator(): ...
def all_timeseries_index_generator(k: int = ...) -> Iterable[Index]:
    """
    Generator which can be iterated over to get instances of all the classes
    which represent time-series.

    Parameters
    ----------
    k: length of each of the index instances
    """
    ...

def makeFloatSeries(name=...): ...
def makeStringSeries(name=...): ...
def makeObjectSeries(name=...): ...
def getSeriesData(): ...
def makeTimeSeries(nper=..., freq=..., name=...): ...
def makePeriodSeries(nper=..., name=...): ...
def getTimeSeriesData(nper=..., freq=...): ...
def getPeriodData(nper=...): ...
def makeTimeDataFrame(nper=..., freq=...): ...
def makeDataFrame() -> DataFrame: ...
def getMixedTypeDict(): ...
def makeMixedDataFrame(): ...
def makePeriodFrame(nper=...): ...
def makeCustomIndex(nentries, nlevels, prefix=..., names=..., ndupe_l=..., idx_type=...):
    """
    Create an index/multindex with given dimensions, levels, names, etc'

    nentries - number of entries in index
    nlevels - number of levels (> 1 produces multindex)
    prefix - a string prefix for labels
    names - (Optional), bool or list of strings. if True will use default
       names, if false will use no names, if a list is given, the name of
       each level in the index will be taken from the list.
    ndupe_l - (Optional), list of ints, the number of rows for which the
       label will repeated at the corresponding level, you can specify just
       the first few, the rest will use the default ndupe_l of 1.
       len(ndupe_l) <= nlevels.
    idx_type - "i"/"f"/"s"/"u"/"dt"/"p"/"td".
       If idx_type is not None, `idx_nlevels` must be 1.
       "i"/"f" creates an integer/float index,
       "s"/"u" creates a string/unicode index
       "dt" create a datetime index.
       "td" create a datetime index.

        if unspecified, string labels will be generated.
    """
    ...

def makeCustomDataframe(
    nrows,
    ncols,
    c_idx_names=...,
    r_idx_names=...,
    c_idx_nlevels=...,
    r_idx_nlevels=...,
    data_gen_f=...,
    c_ndupe_l=...,
    r_ndupe_l=...,
    dtype=...,
    c_idx_type=...,
    r_idx_type=...,
):  # -> DataFrame:
    """
    Create a DataFrame using supplied parameters.

    Parameters
    ----------
    nrows,  ncols - number of data rows/cols
    c_idx_names, idx_names  - False/True/list of strings,  yields No names ,
            default names or uses the provided names for the levels of the
            corresponding index. You can provide a single string when
            c_idx_nlevels ==1.
    c_idx_nlevels - number of levels in columns index. > 1 will yield MultiIndex
    r_idx_nlevels - number of levels in rows index. > 1 will yield MultiIndex
    data_gen_f - a function f(row,col) which return the data value
            at that position, the default generator used yields values of the form
            "RxCy" based on position.
    c_ndupe_l, r_ndupe_l - list of integers, determines the number
            of duplicates for each label at a given level of the corresponding
            index. The default `None` value produces a multiplicity of 1 across
            all levels, i.e. a unique index. Will accept a partial list of length
            N < idx_nlevels, for just the first N levels. If ndupe doesn't divide
            nrows/ncol, the last label might have lower multiplicity.
    dtype - passed to the DataFrame constructor as is, in case you wish to
            have more control in conjunction with a custom `data_gen_f`
    r_idx_type, c_idx_type -  "i"/"f"/"s"/"u"/"dt"/"td".
        If idx_type is not None, `idx_nlevels` must be 1.
        "i"/"f" creates an integer/float index,
        "s"/"u" creates a string/unicode index
        "dt" create a datetime index.
        "td" create a timedelta index.

            if unspecified, string labels will be generated.

    Examples
    --------
    # 5 row, 3 columns, default names on both, single index on both axis
    >> makeCustomDataframe(5,3)

    # make the data a random int between 1 and 100
    >> mkdf(5,3,data_gen_f=lambda r,c:randint(1,100))

    # 2-level multiindex on rows with each label duplicated
    # twice on first level, default names on both axis, single
    # index on both axis
    >> a=makeCustomDataframe(5,3,r_idx_nlevels=2,r_ndupe_l=[2])

    # DatetimeIndex on row, index with unicode labels on columns
    # no names on either axis
    >> a=makeCustomDataframe(5,3,c_idx_names=False,r_idx_names=False,
                             r_idx_type="dt",c_idx_type="u")

    # 4-level multindex on rows with names provided, 2-level multindex
    # on columns with default labels and default names.
    >> a=makeCustomDataframe(5,3,r_idx_nlevels=4,
                             r_idx_names=["FEE","FIH","FOH","FUM"],
                             c_idx_nlevels=2)

    >> a=mkdf(5,3,r_idx_nlevels=2,c_idx_nlevels=4)
    """
    ...

def makeMissingDataframe(density=..., random_state=...): ...
def test_parallel(
    num_threads=..., kwargs_list=...
):  # -> (func: Unknown) -> (*args: Unknown, **kwargs: Unknown) -> None:
    """
    Decorator to run the same function multiple times in parallel.

    Parameters
    ----------
    num_threads : int, optional
        The number of times the function is run in parallel.
    kwargs_list : list of dicts, optional
        The list of kwargs to update original
        function kwargs on different threads.

    Notes
    -----
    This decorator does not pass the return value of the decorated function.

    Original from scikit-image:

    https://github.com/scikit-image/scikit-image/pull/1519

    """
    ...

class SubclassedSeries(Series):
    _metadata = ...

class SubclassedDataFrame(DataFrame):
    _metadata = ...

class SubclassedCategorical(Categorical): ...

def convert_rows_list_to_csv_str(rows_list: list[str]):  # -> str:
    """
    Convert list of CSV rows to single CSV-formatted string for current OS.

    This method is used for creating expected value of to_csv() method.

    Parameters
    ----------
    rows_list : List[str]
        Each element represents the row of csv.

    Returns
    -------
    str
        Expected output of to_csv() in current OS.
    """
    ...

def external_error_raised(expected_exception: type[Exception]) -> ContextManager:
    """
    Helper function to mark pytest.raises that have an external error message.

    Parameters
    ----------
    expected_exception : Exception
        Expected error to raise.

    Returns
    -------
    Callable
        Regular `pytest.raises` function with `match` equal to `None`.
    """
    ...

cython_table = ...

def get_cython_table_params(ndframe, func_names_and_expected):  # -> list[Unknown]:
    """
    Combine frame, functions from com._cython_table
    keys and expected result.

    Parameters
    ----------
    ndframe : DataFrame or Series
    func_names_and_expected : Sequence of two items
        The first item is a name of a NDFrame method ('sum', 'prod') etc.
        The second item is the expected return value.

    Returns
    -------
    list
        List of three items (DataFrame, function, expected result)
    """
    ...

def get_op_from_name(op_name: str) -> Callable:
    """
    The operator function for a given op name.

    Parameters
    ----------
    op_name : str
        The op name, in form of "add" or "__add__".

    Returns
    -------
    function
        A function performing the operation.
    """
    ...

def getitem(x): ...
def setitem(x): ...
def loc(x): ...
def iloc(x): ...
def at(x): ...
def iat(x): ...
