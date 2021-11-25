from pandas import Index
from pandas._libs.lib import NoDefault

def assert_almost_equal(
    left,
    right,
    check_dtype: bool | str = ...,
    check_less_precise: bool | int | NoDefault = ...,
    rtol: float = ...,
    atol: float = ...,
    **kwargs
):  # -> None:
    """
    Check that the left and right objects are approximately equal.

    By approximately equal, we refer to objects that are numbers or that
    contain numbers which may be equivalent to specific levels of precision.

    Parameters
    ----------
    left : object
    right : object
    check_dtype : bool or {'equiv'}, default 'equiv'
        Check dtype if both a and b are the same type. If 'equiv' is passed in,
        then `RangeIndex` and `Int64Index` are also considered equivalent
        when doing type checking.
    check_less_precise : bool or int, default False
        Specify comparison precision. 5 digits (False) or 3 digits (True)
        after decimal points are compared. If int, then specify the number
        of digits to compare.

        When comparing two numbers, if the first number has magnitude less
        than 1e-5, we compare the two numbers directly and check whether
        they are equivalent within the specified precision. Otherwise, we
        compare the **ratio** of the second number to the first number and
        check whether it is equivalent to 1 within the specified precision.

        .. deprecated:: 1.1.0
           Use `rtol` and `atol` instead to define relative/absolute
           tolerance, respectively. Similar to :func:`math.isclose`.
    rtol : float, default 1e-5
        Relative tolerance.

        .. versionadded:: 1.1.0
    atol : float, default 1e-8
        Absolute tolerance.

        .. versionadded:: 1.1.0
    """
    ...

def assert_dict_equal(left, right, compare_keys: bool = ...): ...
def assert_index_equal(
    left: Index,
    right: Index,
    exact: bool | str = ...,
    check_names: bool = ...,
    check_less_precise: bool | int | NoDefault = ...,
    check_exact: bool = ...,
    check_categorical: bool = ...,
    check_order: bool = ...,
    rtol: float = ...,
    atol: float = ...,
    obj: str = ...,
) -> None:
    """
    Check that left and right Index are equal.

    Parameters
    ----------
    left : Index
    right : Index
    exact : bool or {'equiv'}, default 'equiv'
        Whether to check the Index class, dtype and inferred_type
        are identical. If 'equiv', then RangeIndex can be substituted for
        Int64Index as well.
    check_names : bool, default True
        Whether to check the names attribute.
    check_less_precise : bool or int, default False
        Specify comparison precision. Only used when check_exact is False.
        5 digits (False) or 3 digits (True) after decimal points are compared.
        If int, then specify the digits to compare.

        .. deprecated:: 1.1.0
           Use `rtol` and `atol` instead to define relative/absolute
           tolerance, respectively. Similar to :func:`math.isclose`.
    check_exact : bool, default True
        Whether to compare number exactly.
    check_categorical : bool, default True
        Whether to compare internal Categorical exactly.
    check_order : bool, default True
        Whether to compare the order of index entries as well as their values.
        If True, both indexes must contain the same elements, in the same order.
        If False, both indexes must contain the same elements, but in any order.

        .. versionadded:: 1.2.0
    rtol : float, default 1e-5
        Relative tolerance. Only used when check_exact is False.

        .. versionadded:: 1.1.0
    atol : float, default 1e-8
        Absolute tolerance. Only used when check_exact is False.

        .. versionadded:: 1.1.0
    obj : str, default 'Index'
        Specify object name being compared, internally used to show appropriate
        assertion message.

    Examples
    --------
    >>> from pandas.testing import assert_index_equal
    >>> a = pd.Index([1, 2, 3])
    >>> b = pd.Index([1, 2, 3])
    >>> assert_index_equal(a, b)
    """
    ...

def assert_class_equal(left, right, exact: bool | str = ..., obj=...):  # -> None:
    """
    Checks classes are equal.
    """
    ...

def assert_attr_equal(attr: str, left, right, obj: str = ...):  # -> Literal[True]:
    """
    Check attributes are equal. Both objects must have attribute.

    Parameters
    ----------
    attr : str
        Attribute name being compared.
    left : object
    right : object
    obj : str, default 'Attributes'
        Specify object name being compared, internally used to show appropriate
        assertion message
    """
    ...

def assert_is_valid_plot_return_object(objs): ...
def assert_is_sorted(seq):  # -> None:
    """Assert that the sequence is sorted."""
    ...

def assert_categorical_equal(
    left, right, check_dtype=..., check_category_order=..., obj=...
):  # -> None:
    """
    Test that Categoricals are equivalent.

    Parameters
    ----------
    left : Categorical
    right : Categorical
    check_dtype : bool, default True
        Check that integer dtype of the codes are the same
    check_category_order : bool, default True
        Whether the order of the categories should be compared, which
        implies identical integer codes.  If False, only the resulting
        values are compared.  The ordered attribute is
        checked regardless.
    obj : str, default 'Categorical'
        Specify object name being compared, internally used to show appropriate
        assertion message
    """
    ...

def assert_interval_array_equal(left, right, exact=..., obj=...):  # -> None:
    """
    Test that two IntervalArrays are equivalent.

    Parameters
    ----------
    left, right : IntervalArray
        The IntervalArrays to compare.
    exact : bool or {'equiv'}, default 'equiv'
        Whether to check the Index class, dtype and inferred_type
        are identical. If 'equiv', then RangeIndex can be substituted for
        Int64Index as well.
    obj : str, default 'IntervalArray'
        Specify object name being compared, internally used to show appropriate
        assertion message
    """
    ...

def assert_period_array_equal(left, right, obj=...): ...
def assert_datetime_array_equal(left, right, obj=..., check_freq=...): ...
def assert_timedelta_array_equal(left, right, obj=..., check_freq=...): ...
def raise_assert_detail(obj, message, left, right, diff=..., index_values=...): ...
def assert_numpy_array_equal(
    left,
    right,
    strict_nan=...,
    check_dtype=...,
    err_msg=...,
    check_same=...,
    obj=...,
    index_values=...,
):  # -> None:
    """
    Check that 'np.ndarray' is equivalent.

    Parameters
    ----------
    left, right : numpy.ndarray or iterable
        The two arrays to be compared.
    strict_nan : bool, default False
        If True, consider NaN and None to be different.
    check_dtype : bool, default True
        Check dtype if both a and b are np.ndarray.
    err_msg : str, default None
        If provided, used as assertion message.
    check_same : None|'copy'|'same', default None
        Ensure left and right refer/do not refer to the same memory area.
    obj : str, default 'numpy array'
        Specify object name being compared, internally used to show appropriate
        assertion message.
    index_values : numpy.ndarray, default None
        optional index (shared by both left and right), used in output.
    """
    ...

def assert_extension_array_equal(
    left,
    right,
    check_dtype=...,
    index_values=...,
    check_less_precise=...,
    check_exact=...,
    rtol: float = ...,
    atol: float = ...,
):  # -> None:
    """
    Check that left and right ExtensionArrays are equal.

    Parameters
    ----------
    left, right : ExtensionArray
        The two arrays to compare.
    check_dtype : bool, default True
        Whether to check if the ExtensionArray dtypes are identical.
    index_values : numpy.ndarray, default None
        Optional index (shared by both left and right), used in output.
    check_less_precise : bool or int, default False
        Specify comparison precision. Only used when check_exact is False.
        5 digits (False) or 3 digits (True) after decimal points are compared.
        If int, then specify the digits to compare.

        .. deprecated:: 1.1.0
           Use `rtol` and `atol` instead to define relative/absolute
           tolerance, respectively. Similar to :func:`math.isclose`.
    check_exact : bool, default False
        Whether to compare number exactly.
    rtol : float, default 1e-5
        Relative tolerance. Only used when check_exact is False.

        .. versionadded:: 1.1.0
    atol : float, default 1e-8
        Absolute tolerance. Only used when check_exact is False.

        .. versionadded:: 1.1.0

    Notes
    -----
    Missing values are checked separately from valid values.
    A mask of missing values is computed for each and checked to match.
    The remaining all-valid values are cast to object dtype and checked.

    Examples
    --------
    >>> from pandas.testing import assert_extension_array_equal
    >>> a = pd.Series([1, 2, 3, 4])
    >>> b, c = a.array, a.array
    >>> assert_extension_array_equal(b, c)
    """
    ...

def assert_series_equal(
    left,
    right,
    check_dtype=...,
    check_index_type=...,
    check_series_type=...,
    check_less_precise=...,
    check_names=...,
    check_exact=...,
    check_datetimelike_compat=...,
    check_categorical=...,
    check_category_order=...,
    check_freq=...,
    check_flags=...,
    rtol=...,
    atol=...,
    obj=...,
    *,
    check_index=...
):
    """
    Check that left and right Series are equal.

    Parameters
    ----------
    left : Series
    right : Series
    check_dtype : bool, default True
        Whether to check the Series dtype is identical.
    check_index_type : bool or {'equiv'}, default 'equiv'
        Whether to check the Index class, dtype and inferred_type
        are identical.
    check_series_type : bool, default True
         Whether to check the Series class is identical.
    check_less_precise : bool or int, default False
        Specify comparison precision. Only used when check_exact is False.
        5 digits (False) or 3 digits (True) after decimal points are compared.
        If int, then specify the digits to compare.

        When comparing two numbers, if the first number has magnitude less
        than 1e-5, we compare the two numbers directly and check whether
        they are equivalent within the specified precision. Otherwise, we
        compare the **ratio** of the second number to the first number and
        check whether it is equivalent to 1 within the specified precision.

        .. deprecated:: 1.1.0
           Use `rtol` and `atol` instead to define relative/absolute
           tolerance, respectively. Similar to :func:`math.isclose`.
    check_names : bool, default True
        Whether to check the Series and Index names attribute.
    check_exact : bool, default False
        Whether to compare number exactly.
    check_datetimelike_compat : bool, default False
        Compare datetime-like which is comparable ignoring dtype.
    check_categorical : bool, default True
        Whether to compare internal Categorical exactly.
    check_category_order : bool, default True
        Whether to compare category order of internal Categoricals.

        .. versionadded:: 1.0.2
    check_freq : bool, default True
        Whether to check the `freq` attribute on a DatetimeIndex or TimedeltaIndex.

        .. versionadded:: 1.1.0
    check_flags : bool, default True
        Whether to check the `flags` attribute.

        .. versionadded:: 1.2.0

    rtol : float, default 1e-5
        Relative tolerance. Only used when check_exact is False.

        .. versionadded:: 1.1.0
    atol : float, default 1e-8
        Absolute tolerance. Only used when check_exact is False.

        .. versionadded:: 1.1.0
    obj : str, default 'Series'
        Specify object name being compared, internally used to show appropriate
        assertion message.
    check_index : bool, default True
        Whether to check index equivalence. If False, then compare only values.

        .. versionadded:: 1.3.0

    Examples
    --------
    >>> from pandas.testing import assert_series_equal
    >>> a = pd.Series([1, 2, 3, 4])
    >>> b = pd.Series([1, 2, 3, 4])
    >>> assert_series_equal(a, b)
    """
    ...

def assert_frame_equal(
    left,
    right,
    check_dtype=...,
    check_index_type=...,
    check_column_type=...,
    check_frame_type=...,
    check_less_precise=...,
    check_names=...,
    by_blocks=...,
    check_exact=...,
    check_datetimelike_compat=...,
    check_categorical=...,
    check_like=...,
    check_freq=...,
    check_flags=...,
    rtol=...,
    atol=...,
    obj=...,
):
    """
    Check that left and right DataFrame are equal.

    This function is intended to compare two DataFrames and output any
    differences. Is is mostly intended for use in unit tests.
    Additional parameters allow varying the strictness of the
    equality checks performed.

    Parameters
    ----------
    left : DataFrame
        First DataFrame to compare.
    right : DataFrame
        Second DataFrame to compare.
    check_dtype : bool, default True
        Whether to check the DataFrame dtype is identical.
    check_index_type : bool or {'equiv'}, default 'equiv'
        Whether to check the Index class, dtype and inferred_type
        are identical.
    check_column_type : bool or {'equiv'}, default 'equiv'
        Whether to check the columns class, dtype and inferred_type
        are identical. Is passed as the ``exact`` argument of
        :func:`assert_index_equal`.
    check_frame_type : bool, default True
        Whether to check the DataFrame class is identical.
    check_less_precise : bool or int, default False
        Specify comparison precision. Only used when check_exact is False.
        5 digits (False) or 3 digits (True) after decimal points are compared.
        If int, then specify the digits to compare.

        When comparing two numbers, if the first number has magnitude less
        than 1e-5, we compare the two numbers directly and check whether
        they are equivalent within the specified precision. Otherwise, we
        compare the **ratio** of the second number to the first number and
        check whether it is equivalent to 1 within the specified precision.

        .. deprecated:: 1.1.0
           Use `rtol` and `atol` instead to define relative/absolute
           tolerance, respectively. Similar to :func:`math.isclose`.
    check_names : bool, default True
        Whether to check that the `names` attribute for both the `index`
        and `column` attributes of the DataFrame is identical.
    by_blocks : bool, default False
        Specify how to compare internal data. If False, compare by columns.
        If True, compare by blocks.
    check_exact : bool, default False
        Whether to compare number exactly.
    check_datetimelike_compat : bool, default False
        Compare datetime-like which is comparable ignoring dtype.
    check_categorical : bool, default True
        Whether to compare internal Categorical exactly.
    check_like : bool, default False
        If True, ignore the order of index & columns.
        Note: index labels must match their respective rows
        (same as in columns) - same labels must be with the same data.
    check_freq : bool, default True
        Whether to check the `freq` attribute on a DatetimeIndex or TimedeltaIndex.

        .. versionadded:: 1.1.0
    check_flags : bool, default True
        Whether to check the `flags` attribute.
    rtol : float, default 1e-5
        Relative tolerance. Only used when check_exact is False.

        .. versionadded:: 1.1.0
    atol : float, default 1e-8
        Absolute tolerance. Only used when check_exact is False.

        .. versionadded:: 1.1.0
    obj : str, default 'DataFrame'
        Specify object name being compared, internally used to show appropriate
        assertion message.

    See Also
    --------
    assert_series_equal : Equivalent method for asserting Series equality.
    DataFrame.equals : Check DataFrame equality.

    Examples
    --------
    This example shows comparing two DataFrames that are equal
    but with columns of differing dtypes.

    >>> from pandas._testing import assert_frame_equal
    >>> df1 = pd.DataFrame({'a': [1, 2], 'b': [3, 4]})
    >>> df2 = pd.DataFrame({'a': [1, 2], 'b': [3.0, 4.0]})

    df1 equals itself.

    >>> assert_frame_equal(df1, df1)

    df1 differs from df2 as column 'b' is of a different type.

    >>> assert_frame_equal(df1, df2)
    Traceback (most recent call last):
    ...
    AssertionError: Attributes of DataFrame.iloc[:, 1] (column name="b") are different

    Attribute "dtype" are different
    [left]:  int64
    [right]: float64

    Ignore differing dtypes in columns with check_dtype.

    >>> assert_frame_equal(df1, df2, check_dtype=False)
    """
    ...

def assert_equal(left, right, **kwargs):
    """
    Wrapper for tm.assert_*_equal to dispatch to the appropriate test function.

    Parameters
    ----------
    left, right : Index, Series, DataFrame, ExtensionArray, or np.ndarray
        The two items to be compared.
    **kwargs
        All keyword arguments are passed through to the underlying assert method.
    """
    ...

def assert_sp_array_equal(left, right):  # -> None:
    """
    Check that the left and right SparseArray are equal.

    Parameters
    ----------
    left : SparseArray
    right : SparseArray
    """
    ...

def assert_contains_all(iterable, dic): ...
def assert_copy(iter1, iter2, **eql_kwargs):  # -> None:
    """
    iter1, iter2: iterables that produce elements
    comparable with assert_almost_equal

    Checks that the elements are equal, but not
    the same object. (Does not check that items
    in sequences are also not the same object)
    """
    ...

def is_extension_array_dtype_and_needs_i8_conversion(left_dtype, right_dtype) -> bool:
    """
    Checks that we have the combination of an ExtensionArraydtype and
    a dtype that should be converted to int64

    Returns
    -------
    bool

    Related to issue #37609
    """
    ...
