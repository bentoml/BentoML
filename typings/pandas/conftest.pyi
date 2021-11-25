

import operator
from collections import abc
from datetime import timezone

import numpy as np
import pandas as pd
import pandas._testing as tm
import pandas.util._test_decorators as td
import pytest
from dateutil.tz import tzutc
from pandas import DataFrame, Interval, Period, Series, Timedelta, Timestamp
from pandas.core import ops
from pandas.core.dtypes.dtypes import DatetimeTZDtype, IntervalDtype
from pandas.core.indexes.api import Index, MultiIndex
from pytz import utc

"""
This file is very long and growing, but it was decided to not split it yet, as
it's still manageable (2020-03-17, ~1.1k LoC). See gh-31989

Instead of splitting it was decided to define sections here:
- Configuration / Settings
- Autouse fixtures
- Common arguments
- Missing values & co.
- Classes
- Indices
- Series'
- DataFrames
- Operators & Operations
- Data sets/files
- Time zones
- Dtypes
- Misc
"""
suppress_npdev_promotion_warning = ...
def pytest_addoption(parser): # -> None:
    ...

def pytest_runtest_setup(item): # -> None:
    ...

def pytest_collection_modifyitems(items): # -> None:
    ...

@pytest.fixture(autouse=True)
def configure_tests(): # -> None:
    """
    Configure settings for all tests and test modules.
    """
    ...

@pytest.fixture(autouse=True)
def add_imports(doctest_namespace): # -> None:
    """
    Make `np` and `pd` names available for doctests.
    """
    ...

@pytest.fixture(params=[0, 1, "index", "columns"], ids=lambda x: <Expression>)
def axis(request):
    """
    Fixture for returning the axis numbers of a DataFrame.
    """
    ...

axis_frame = ...
@pytest.fixture(params=[True, False, None])
def observed(request):
    """
    Pass in the observed keyword to groupby for [True, False]
    This indicates whether categoricals should return values for
    values which are not in the grouper [False / None], or only values which
    appear in the grouper [True]. [None] is supported for future compatibility
    if we decide to change the default (and would need to warn if this
    parameter is not passed).
    """
    ...

@pytest.fixture(params=[True, False, None])
def ordered(request):
    """
    Boolean 'ordered' parameter for Categorical.
    """
    ...

@pytest.fixture(params=["first", "last", False])
def keep(request):
    """
    Valid values for the 'keep' parameter used in
    .duplicated or .drop_duplicates
    """
    ...

@pytest.fixture(params=["left", "right", "both", "neither"])
def closed(request):
    """
    Fixture for trying all interval closed parameters.
    """
    ...

@pytest.fixture(params=["left", "right", "both", "neither"])
def other_closed(request):
    """
    Secondary closed fixture to allow parametrizing over all pairs of closed.
    """
    ...

@pytest.fixture(params=[None, "gzip", "bz2", "zip", "xz"])
def compression(request):
    """
    Fixture for trying common compression types in compression tests.
    """
    ...

@pytest.fixture(params=["gzip", "bz2", "zip", "xz"])
def compression_only(request):
    """
    Fixture for trying common compression types in compression tests excluding
    uncompressed case.
    """
    ...

@pytest.fixture(params=[True, False])
def writable(request):
    """
    Fixture that an array is writable.
    """
    ...

@pytest.fixture(params=["inner", "outer", "left", "right"])
def join_type(request):
    """
    Fixture for trying all types of join operations.
    """
    ...

@pytest.fixture(params=["nlargest", "nsmallest"])
def nselect_method(request):
    """
    Fixture for trying all nselect methods.
    """
    ...

@pytest.fixture(params=tm.NULL_OBJECTS, ids=lambda x: type(x).__name__)
def nulls_fixture(request):
    """
    Fixture for each null type in pandas.
    """
    ...

nulls_fixture2 = ...
@pytest.fixture(params=[None, np.nan, pd.NaT])
def unique_nulls_fixture(request):
    """
    Fixture for each null type in pandas, each null type exactly once.
    """
    ...

unique_nulls_fixture2 = ...
@pytest.fixture(params=[DataFrame, Series])
def frame_or_series(request):
    """
    Fixture to parametrize over DataFrame and Series.
    """
    ...

@pytest.fixture(params=[Index, Series], ids=["index", "series"])
def index_or_series(request):
    """
    Fixture to parametrize over Index and Series, made necessary by a mypy
    bug, giving an error:

    List item 0 has incompatible type "Type[Series]"; expected "Type[PandasObject]"

    See GH#29725
    """
    ...

index_or_series2 = ...
@pytest.fixture(params=[Index, Series, pd.array], ids=["index", "series", "array"])
def index_or_series_or_array(request):
    """
    Fixture to parametrize over Index, Series, and ExtensionArray
    """
    ...

@pytest.fixture
def dict_subclass(): # -> Type[TestSubDict]:
    """
    Fixture for a dictionary subclass.
    """
    class TestSubDict(dict):
        ...
    
    

@pytest.fixture
def non_dict_mapping_subclass(): # -> Type[TestNonDictMapping]:
    """
    Fixture for a non-mapping dictionary subclass.
    """
    class TestNonDictMapping(abc.Mapping):
        ...
    
    

@pytest.fixture
def multiindex_year_month_day_dataframe_random_data(): # -> NoReturn:
    """
    DataFrame with 3 level MultiIndex (year, month, day) covering
    first 100 business days from 2000-01-01 with random data
    """
    ...

@pytest.fixture
def multiindex_dataframe_random_data(): # -> DataFrame:
    """DataFrame with 2 level MultiIndex with random data"""
    ...

indices_dict = ...
@pytest.fixture(params=indices_dict.keys())
def index(request): # -> Index | DatetimeIndex | PeriodIndex | TimedeltaIndex | Int64Index | UInt64Index | NumericIndex | RangeIndex | Float64Index | CategoricalIndex | IntervalIndex | MultiIndex:
    """
    Fixture for many "simple" kinds of indices.

    These indices are unlikely to cover corner cases, e.g.
        - no names
        - no NaTs/NaNs
        - no values near implementation bounds
        - ...
    """
    ...

index_fixture2 = ...
@pytest.fixture(params=[key for key in indices_dict if notisinstance(indices_dict[key], MultiIndex)])
def index_flat(request): # -> Index | DatetimeIndex | PeriodIndex | TimedeltaIndex | Int64Index | UInt64Index | NumericIndex | RangeIndex | Float64Index | CategoricalIndex | IntervalIndex | MultiIndex:
    """
    index fixture, but excluding MultiIndex cases.
    """
    ...

index_flat2 = ...
@pytest.fixture(params=[key for key in indices_dict if key not in ["int", "uint", "range", "empty", "repeats"] and notisinstance(indices_dict[key], MultiIndex)])
def index_with_missing(request): # -> MultiIndex | Any:
    """
    Fixture for indices with missing values.

    Integer-dtype and empty cases are excluded because they cannot hold missing
    values.

    MultiIndex is excluded because isna() is not defined for MultiIndex.
    """
    ...

@pytest.fixture
def empty_series(): # -> Series:
    ...

@pytest.fixture
def string_series(): # -> Series:
    """
    Fixture for Series of floats with Index of unique strings
    """
    ...

@pytest.fixture
def object_series(): # -> Series:
    """
    Fixture for Series of dtype object with Index of unique strings
    """
    ...

@pytest.fixture
def datetime_series(): # -> Series:
    """
    Fixture for Series of floats with DatetimeIndex
    """
    ...

_series = ...
@pytest.fixture
def series_with_simple_index(index): # -> Series:
    """
    Fixture for tests on series with changing types of indices.
    """
    ...

@pytest.fixture
def series_with_multilevel_index(): # -> Series:
    """
    Fixture with a Series with a 2-level MultiIndex.
    """
    ...

_narrow_dtypes = ...
_narrow_series = ...
@pytest.fixture(params=_narrow_series.keys())
def narrow_series(request): # -> Series:
    """
    Fixture for Series with low precision data types
    """
    ...

_index_or_series_objs = ...
@pytest.fixture(params=_index_or_series_objs.keys())
def index_or_series_obj(request): # -> Index | DatetimeIndex | PeriodIndex | TimedeltaIndex | Int64Index | UInt64Index | NumericIndex | RangeIndex | Float64Index | CategoricalIndex | IntervalIndex | MultiIndex | Series:
    """
    Fixture for tests on indexes, series and series with a narrow dtype
    copy to avoid mutation, e.g. setting .name
    """
    ...

@pytest.fixture
def empty_frame(): # -> DataFrame:
    ...

@pytest.fixture
def int_frame(): # -> DataFrame:
    """
    Fixture for DataFrame of ints with index of unique strings

    Columns are ['A', 'B', 'C', 'D']

                A  B  C  D
    vpBeWjM651  1  0  1  0
    5JyxmrP1En -1  0  0  0
    qEDaoD49U2 -1  1  0  0
    m66TkTfsFe  0  0  0  0
    EHPaNzEUFm -1  0 -1  0
    fpRJCevQhi  2  0  0  0
    OlQvnmfi3Q  0  0 -2  0
    ...        .. .. .. ..
    uB1FPlz4uP  0  0  0  1
    EcSe6yNzCU  0  0 -1  0
    L50VudaiI8 -1  1 -2  0
    y3bpw4nwIp  0 -1  0  0
    H0RdLLwrCT  1  1  0  0
    rY82K0vMwm  0  0  0  0
    1OPIUjnkjk  2  0  0  0

    [30 rows x 4 columns]
    """
    ...

@pytest.fixture
def datetime_frame(): # -> DataFrame:
    """
    Fixture for DataFrame of floats with DatetimeIndex

    Columns are ['A', 'B', 'C', 'D']

                       A         B         C         D
    2000-01-03 -1.122153  0.468535  0.122226  1.693711
    2000-01-04  0.189378  0.486100  0.007864 -1.216052
    2000-01-05  0.041401 -0.835752 -0.035279 -0.414357
    2000-01-06  0.430050  0.894352  0.090719  0.036939
    2000-01-07 -0.620982 -0.668211 -0.706153  1.466335
    2000-01-10 -0.752633  0.328434 -0.815325  0.699674
    2000-01-11 -2.236969  0.615737 -0.829076 -1.196106
    ...              ...       ...       ...       ...
    2000-02-03  1.642618 -0.579288  0.046005  1.385249
    2000-02-04 -0.544873 -1.160962 -0.284071 -1.418351
    2000-02-07 -2.656149 -0.601387  1.410148  0.444150
    2000-02-08 -1.201881 -1.289040  0.772992 -1.445300
    2000-02-09  1.377373  0.398619  1.008453 -0.928207
    2000-02-10  0.473194 -0.636677  0.984058  0.511519
    2000-02-11 -0.965556  0.408313 -1.312844 -0.381948

    [30 rows x 4 columns]
    """
    ...

@pytest.fixture
def float_frame(): # -> DataFrame:
    """
    Fixture for DataFrame of floats with index of unique strings

    Columns are ['A', 'B', 'C', 'D'].

                       A         B         C         D
    P7GACiRnxd -0.465578 -0.361863  0.886172 -0.053465
    qZKh6afn8n -0.466693 -0.373773  0.266873  1.673901
    tkp0r6Qble  0.148691 -0.059051  0.174817  1.598433
    wP70WOCtv8  0.133045 -0.581994 -0.992240  0.261651
    M2AeYQMnCz -1.207959 -0.185775  0.588206  0.563938
    QEPzyGDYDo -0.381843 -0.758281  0.502575 -0.565053
    r78Jwns6dn -0.653707  0.883127  0.682199  0.206159
    ...              ...       ...       ...       ...
    IHEGx9NO0T -0.277360  0.113021 -1.018314  0.196316
    lPMj8K27FA -1.313667 -0.604776 -1.305618 -0.863999
    qa66YMWQa5  1.110525  0.475310 -0.747865  0.032121
    yOa0ATsmcE -0.431457  0.067094  0.096567 -0.264962
    65znX3uRNG  1.528446  0.160416 -0.109635 -0.032987
    eCOBvKqf3e  0.235281  1.622222  0.781255  0.392871
    xSucinXxuV -1.263557  0.252799 -0.552247  0.400426

    [30 rows x 4 columns]
    """
    ...

@pytest.fixture
def mixed_type_frame(): # -> DataFrame:
    """
    Fixture for DataFrame of float/int/string columns with RangeIndex
    Columns are ['a', 'b', 'c', 'float32', 'int32'].
    """
    ...

@pytest.fixture
def rand_series_with_duplicate_datetimeindex(): # -> Series:
    """
    Fixture for Series with a DatetimeIndex that has duplicates.
    """
    ...

@pytest.fixture(params=[(Interval(left=0, right=5), IntervalDtype("int64", "right")), (Interval(left=0.1, right=0.5), IntervalDtype("float64", "right")), (Period("2012-01", freq="M"), "period[M]"), (Period("2012-02-01", freq="D"), "period[D]"), (Timestamp("2011-01-01", tz="US/Eastern"), DatetimeTZDtype(tz="US/Eastern")), (Timedelta(seconds=500), "timedelta64[ns]")])
def ea_scalar_and_dtype(request):
    ...

_all_arithmetic_operators = ...
@pytest.fixture(params=_all_arithmetic_operators)
def all_arithmetic_operators(request):
    """
    Fixture for dunder names for common arithmetic operations.
    """
    ...

@pytest.fixture(params=[operator.add, ops.radd, operator.sub, ops.rsub, operator.mul, ops.rmul, operator.truediv, ops.rtruediv, operator.floordiv, ops.rfloordiv, operator.mod, ops.rmod, operator.pow, ops.rpow, operator.eq, operator.ne, operator.lt, operator.le, operator.gt, operator.ge, operator.and_, ops.rand_, operator.xor, ops.rxor, operator.or_, ops.ror_])
def all_binary_operators(request):
    """
    Fixture for operator and roperator arithmetic, comparison, and logical ops.
    """
    ...

@pytest.fixture(params=[operator.add, ops.radd, operator.sub, ops.rsub, operator.mul, ops.rmul, operator.truediv, ops.rtruediv, operator.floordiv, ops.rfloordiv, operator.mod, ops.rmod, operator.pow, ops.rpow])
def all_arithmetic_functions(request):
    """
    Fixture for operator and roperator arithmetic functions.

    Notes
    -----
    This includes divmod and rdivmod, whereas all_arithmetic_operators
    does not.
    """
    ...

_all_numeric_reductions = ...
@pytest.fixture(params=_all_numeric_reductions)
def all_numeric_reductions(request):
    """
    Fixture for numeric reduction names.
    """
    ...

_all_boolean_reductions = ...
@pytest.fixture(params=_all_boolean_reductions)
def all_boolean_reductions(request):
    """
    Fixture for boolean reduction names.
    """
    ...

_all_reductions = ...
@pytest.fixture(params=_all_reductions)
def all_reductions(request):
    """
    Fixture for all (boolean + numeric) reduction names.
    """
    ...

@pytest.fixture(params=["__eq__", "__ne__", "__le__", "__lt__", "__ge__", "__gt__"])
def all_compare_operators(request):
    """
    Fixture for dunder names for common compare operations

    * >=
    * >
    * ==
    * !=
    * <
    * <=
    """
    ...

@pytest.fixture(params=["__le__", "__lt__", "__ge__", "__gt__"])
def compare_operators_no_eq_ne(request):
    """
    Fixture for dunder names for compare operations except == and !=

    * >=
    * >
    * <
    * <=
    """
    ...

@pytest.fixture(params=["__and__", "__rand__", "__or__", "__ror__", "__xor__", "__rxor__"])
def all_logical_operators(request):
    """
    Fixture for dunder names for common logical operations

    * |
    * &
    * ^
    """
    ...

@pytest.fixture
def strict_data_files(pytestconfig):
    """
    Returns the configuration for the test setting `--strict-data-files`.
    """
    ...

@pytest.fixture
def datapath(strict_data_files): # -> (*args: Unknown) -> str:
    """
    Get the path to a data file.

    Parameters
    ----------
    path : str
        Path to the file, relative to ``pandas/tests/``

    Returns
    -------
    path including ``pandas/tests``.

    Raises
    ------
    ValueError
        If the path doesn't exist and the --strict-data-files option is set.
    """
    ...

@pytest.fixture
def iris(datapath): # -> TextFileReader | DataFrame:
    """
    The iris dataset as a DataFrame.
    """
    ...

TIMEZONES = ...
TIMEZONE_IDS = ...
@td.parametrize_fixture_doc(str(TIMEZONE_IDS))
@pytest.fixture(params=TIMEZONES, ids=TIMEZONE_IDS)
def tz_naive_fixture(request):
    """
    Fixture for trying timezones including default (None): {0}
    """
    ...

@td.parametrize_fixture_doc(str(TIMEZONE_IDS[1]))
@pytest.fixture(params=TIMEZONES[1], ids=TIMEZONE_IDS[1])
def tz_aware_fixture(request):
    """
    Fixture for trying explicit timezones: {0}
    """
    ...

tz_aware_fixture2 = ...
@pytest.fixture(params=["utc", "dateutil/UTC", utc, tzutc(), timezone.utc])
def utc_fixture(request):
    """
    Fixture to provide variants of UTC timezone strings and tzinfo objects.
    """
    ...

utc_fixture2 = ...
@pytest.fixture(params=tm.STRING_DTYPES)
def string_dtype(request):
    """
    Parametrized fixture for string dtypes.

    * str
    * 'str'
    * 'U'
    """
    ...

@pytest.fixture(params=["string[python]", pytest.param("string[pyarrow]", marks=td.skip_if_no("pyarrow", min_version="1.0.0"))])
def nullable_string_dtype(request):
    """
    Parametrized fixture for string dtypes.

    * 'string[python]'
    * 'string[pyarrow]'
    """
    ...

@pytest.fixture(params=["python", pytest.param("pyarrow", marks=td.skip_if_no("pyarrow", min_version="1.0.0"))])
def string_storage(request):
    """
    Parametrized fixture for pd.options.mode.string_storage.

    * 'python'
    * 'pyarrow'
    """
    ...

string_storage2 = ...
@pytest.fixture(params=tm.BYTES_DTYPES)
def bytes_dtype(request):
    """
    Parametrized fixture for bytes dtypes.

    * bytes
    * 'bytes'
    """
    ...

@pytest.fixture(params=tm.OBJECT_DTYPES)
def object_dtype(request):
    """
    Parametrized fixture for object dtypes.

    * object
    * 'object'
    """
    ...

@pytest.fixture(params=["object", "string[python]", pytest.param("string[pyarrow]", marks=td.skip_if_no("pyarrow", min_version="1.0.0"))])
def any_string_dtype(request):
    """
    Parametrized fixture for string dtypes.
    * 'object'
    * 'string[python]'
    * 'string[pyarrow]'
    """
    ...

@pytest.fixture(params=tm.DATETIME64_DTYPES)
def datetime64_dtype(request):
    """
    Parametrized fixture for datetime64 dtypes.

    * 'datetime64[ns]'
    * 'M8[ns]'
    """
    ...

@pytest.fixture(params=tm.TIMEDELTA64_DTYPES)
def timedelta64_dtype(request):
    """
    Parametrized fixture for timedelta64 dtypes.

    * 'timedelta64[ns]'
    * 'm8[ns]'
    """
    ...

@pytest.fixture(params=tm.FLOAT_DTYPES)
def float_dtype(request):
    """
    Parameterized fixture for float dtypes.

    * float
    * 'float32'
    * 'float64'
    """
    ...

@pytest.fixture(params=tm.FLOAT_EA_DTYPES)
def float_ea_dtype(request):
    """
    Parameterized fixture for float dtypes.

    * 'Float32'
    * 'Float64'
    """
    ...

@pytest.fixture(params=tm.FLOAT_DTYPES + tm.FLOAT_EA_DTYPES)
def any_float_allowed_nullable_dtype(request):
    """
    Parameterized fixture for float dtypes.

    * float
    * 'float32'
    * 'float64'
    * 'Float32'
    * 'Float64'
    """
    ...

@pytest.fixture(params=tm.COMPLEX_DTYPES)
def complex_dtype(request):
    """
    Parameterized fixture for complex dtypes.

    * complex
    * 'complex64'
    * 'complex128'
    """
    ...

@pytest.fixture(params=tm.SIGNED_INT_DTYPES)
def sint_dtype(request):
    """
    Parameterized fixture for signed integer dtypes.

    * int
    * 'int8'
    * 'int16'
    * 'int32'
    * 'int64'
    """
    ...

@pytest.fixture(params=tm.UNSIGNED_INT_DTYPES)
def uint_dtype(request):
    """
    Parameterized fixture for unsigned integer dtypes.

    * 'uint8'
    * 'uint16'
    * 'uint32'
    * 'uint64'
    """
    ...

@pytest.fixture(params=tm.ALL_INT_DTYPES)
def any_int_dtype(request):
    """
    Parameterized fixture for any integer dtype.

    * int
    * 'int8'
    * 'uint8'
    * 'int16'
    * 'uint16'
    * 'int32'
    * 'uint32'
    * 'int64'
    * 'uint64'
    """
    ...

@pytest.fixture(params=tm.ALL_EA_INT_DTYPES)
def any_nullable_int_dtype(request):
    """
    Parameterized fixture for any nullable integer dtype.

    * 'UInt8'
    * 'Int8'
    * 'UInt16'
    * 'Int16'
    * 'UInt32'
    * 'Int32'
    * 'UInt64'
    * 'Int64'
    """
    ...

@pytest.fixture(params=tm.ALL_INT_DTYPES + tm.ALL_EA_INT_DTYPES)
def any_int_or_nullable_int_dtype(request):
    """
    Parameterized fixture for any nullable integer dtype.

    * int
    * 'int8'
    * 'uint8'
    * 'int16'
    * 'uint16'
    * 'int32'
    * 'uint32'
    * 'int64'
    * 'uint64'
    * 'UInt8'
    * 'Int8'
    * 'UInt16'
    * 'Int16'
    * 'UInt32'
    * 'Int32'
    * 'UInt64'
    * 'Int64'
    """
    ...

@pytest.fixture(params=tm.ALL_EA_INT_DTYPES + tm.FLOAT_EA_DTYPES)
def any_nullable_numeric_dtype(request):
    """
    Parameterized fixture for any nullable integer dtype and
    any float ea dtypes.

    * 'UInt8'
    * 'Int8'
    * 'UInt16'
    * 'Int16'
    * 'UInt32'
    * 'Int32'
    * 'UInt64'
    * 'Int64'
    * 'Float32'
    * 'Float64'
    """
    ...

@pytest.fixture(params=tm.SIGNED_EA_INT_DTYPES)
def any_signed_nullable_int_dtype(request):
    """
    Parameterized fixture for any signed nullable integer dtype.

    * 'Int8'
    * 'Int16'
    * 'Int32'
    * 'Int64'
    """
    ...

@pytest.fixture(params=tm.ALL_REAL_DTYPES)
def any_real_dtype(request):
    """
    Parameterized fixture for any (purely) real numeric dtype.

    * int
    * 'int8'
    * 'uint8'
    * 'int16'
    * 'uint16'
    * 'int32'
    * 'uint32'
    * 'int64'
    * 'uint64'
    * float
    * 'float32'
    * 'float64'
    """
    ...

@pytest.fixture(params=tm.ALL_NUMPY_DTYPES)
def any_numpy_dtype(request):
    """
    Parameterized fixture for all numpy dtypes.

    * bool
    * 'bool'
    * int
    * 'int8'
    * 'uint8'
    * 'int16'
    * 'uint16'
    * 'int32'
    * 'uint32'
    * 'int64'
    * 'uint64'
    * float
    * 'float32'
    * 'float64'
    * complex
    * 'complex64'
    * 'complex128'
    * str
    * 'str'
    * 'U'
    * bytes
    * 'bytes'
    * 'datetime64[ns]'
    * 'M8[ns]'
    * 'timedelta64[ns]'
    * 'm8[ns]'
    * object
    * 'object'
    """
    ...

_any_skipna_inferred_dtype = ...
@pytest.fixture(params=_any_skipna_inferred_dtype, ids=ids)
def any_skipna_inferred_dtype(request): # -> tuple[Unknown, ndarray]:
    """
    Fixture for all inferred dtypes from _libs.lib.infer_dtype

    The covered (inferred) types are:
    * 'string'
    * 'empty'
    * 'bytes'
    * 'mixed'
    * 'mixed-integer'
    * 'mixed-integer-float'
    * 'floating'
    * 'integer'
    * 'decimal'
    * 'boolean'
    * 'datetime64'
    * 'datetime'
    * 'date'
    * 'timedelta'
    * 'time'
    * 'period'
    * 'interval'

    Returns
    -------
    inferred_dtype : str
        The string for the inferred dtype from _libs.lib.infer_dtype
    values : np.ndarray
        An array of object dtype that will be inferred to have
        `inferred_dtype`

    Examples
    --------
    >>> import pandas._libs.lib as lib
    >>>
    >>> def test_something(any_skipna_inferred_dtype):
    ...     inferred_dtype, values = any_skipna_inferred_dtype
    ...     # will pass
    ...     assert lib.infer_dtype(values, skipna=True) == inferred_dtype
    """
    ...

@pytest.fixture
def ip():
    """
    Get an instance of IPython.InteractiveShell.

    Will raise a skip if IPython is not installed.
    """
    ...

@pytest.fixture(params=["bsr", "coo", "csc", "csr", "dia", "dok", "lil"])
def spmatrix(request): # -> Any:
    """
    Yields scipy sparse matrix classes.
    """
    ...

@pytest.fixture(params=[getattr(pd.offsets, o) for o in pd.offsets.__all__ if issubclass(getattr(pd.offsets, o), pd.offsets.Tick)])
def tick_classes(request):
    """
    Fixture for Tick based datetime offsets available for a time series.
    """
    ...

@pytest.fixture(params=[None, lambda x: x])
def sort_by_key(request):
    """
    Simple fixture for testing keys in sorting methods.
    Tests None (no key) and the identity key.
    """
    ...

@pytest.fixture()
def fsspectest(): # -> Generator[TestMemoryFS, None, None]:
    class TestMemoryFS(MemoryFileSystem):
        ...
    
    

@pytest.fixture(params=[("foo", None, None), ("Egon", "Venkman", None), ("NCC1701D", "NCC1701D", "NCC1701D")])
def names(request):
    """
    A 3-tuple of names, the first two for operands, the last for a result.
    """
    ...

@pytest.fixture(params=[tm.setitem, tm.loc, tm.iloc])
def indexer_sli(request):
    """
    Parametrize over __setitem__, loc.__setitem__, iloc.__setitem__
    """
    ...

@pytest.fixture(params=[tm.setitem, tm.iloc])
def indexer_si(request):
    """
    Parametrize over __setitem__, iloc.__setitem__
    """
    ...

@pytest.fixture(params=[tm.setitem, tm.loc])
def indexer_sl(request):
    """
    Parametrize over __setitem__, loc.__setitem__
    """
    ...

@pytest.fixture(params=[tm.at, tm.loc])
def indexer_al(request):
    """
    Parametrize over at.__setitem__, loc.__setitem__
    """
    ...

@pytest.fixture
def using_array_manager(request): # -> bool | Any:
    """
    Fixture to check if the array manager is being used.
    """
    ...

