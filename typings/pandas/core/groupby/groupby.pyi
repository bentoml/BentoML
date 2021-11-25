from contextlib import contextmanager
from textwrap import dedent
from typing import (
    TYPE_CHECKING,
    Callable,
    Hashable,
    Iterator,
    List,
    Literal,
    Mapping,
    Sequence,
    Union,
)

import numpy as np
from pandas._libs import lib
from pandas._typing import FrameOrSeries, FrameOrSeriesUnion, IndexLabel, T, final
from pandas.core.base import PandasObject, SelectionMixin
from pandas.core.frame import DataFrame
from pandas.core.generic import NDFrame
from pandas.core.groupby import ops
from pandas.core.series import Series
from pandas.util._decorators import Appender, Substitution, doc

"""
Provide the groupby split-apply-combine paradigm. Define the GroupBy
class providing the base-class of operations.

The SeriesGroupBy and DataFrameGroupBy sub-class
(defined in pandas.core.groupby.generic)
expose these user-facing objects to provide specific functionality.
"""
if TYPE_CHECKING: ...
_common_see_also = ...
_apply_docs = ...
_groupby_agg_method_template = ...
_pipe_template = ...
_transform_template = ...
_agg_template = ...

@final
class GroupByPlot(PandasObject):
    """
    Class implementing the .plot attribute for groupby objects.
    """

    def __init__(self, groupby: GroupBy) -> None: ...
    def __call__(self, *args, **kwargs): ...
    def __getattr__(self, name: str): ...

@contextmanager
def group_selection_context(groupby: GroupBy) -> Iterator[GroupBy]:
    """
    Set / reset the group_selection_context.
    """
    ...

_KeysArgType = (
    Union[
        Hashable,
        List[Hashable],
        Callable[[Hashable], Hashable],
        List[Callable[[Hashable], Hashable]],
        Mapping[Hashable, Hashable],
    ],
)

class BaseGroupBy(PandasObject, SelectionMixin[FrameOrSeries]):
    _group_selection: IndexLabel | None = ...
    _apply_allowlist: frozenset[str] = ...
    _hidden_attrs = ...
    axis: int
    grouper: ops.BaseGrouper
    group_keys: bool
    @final
    def __len__(self) -> int: ...
    @final
    def __repr__(self) -> str: ...
    @final
    @property
    def groups(self) -> dict[Hashable, np.ndarray]:
        """
        Dict {group name -> group labels}.
        """
        ...
    @final
    @property
    def ngroups(self) -> int: ...
    @final
    @property
    def indices(
        self,
    ):  # -> () -> (() -> (() -> Unknown | dict[Hashable, ndarray]) | dict[str | tuple[Unknown, ...], ndarray]):
        """
        Dict {group name -> group indices}.
        """
        ...
    @Substitution(
        klass="GroupBy",
        examples=dedent(
            """\
        >>> df = pd.DataFrame({'A': 'a b a b'.split(), 'B': [1, 2, 3, 4]})
        >>> df
           A  B
        0  a  1
        1  b  2
        2  a  3
        3  b  4

        To get the difference between each groups maximum and minimum value in one
        pass, you can do

        >>> df.groupby('A').pipe(lambda x: x.max() - x.min())
           B
        A
        a  2
        b  2"""
        ),
    )
    @Appender(_pipe_template)
    def pipe(
        self, func: Callable[..., T] | tuple[Callable[..., T], str], *args, **kwargs
    ) -> T: ...
    plot = ...
    @final
    def get_group(self, name, obj=...) -> FrameOrSeriesUnion:
        """
        Construct DataFrame from group with provided name.

        Parameters
        ----------
        name : object
            The name of the group to get as a DataFrame.
        obj : DataFrame, default None
            The DataFrame to take the DataFrame out of.  If
            it is None, the object groupby was called on will
            be used.

        Returns
        -------
        group : same type as obj
        """
        ...
    @final
    def __iter__(self) -> Iterator[tuple[Hashable, FrameOrSeries]]:
        """
        Groupby iterator.

        Returns
        -------
        Generator yielding sequence of (name, subsetted object)
        for each group
        """
        ...

OutputFrameOrSeries = ...

class GroupBy(BaseGroupBy[FrameOrSeries]):
    """
    Class for grouping and aggregating relational data.

    See aggregate, transform, and apply functions on this object.

    It's easiest to use obj.groupby(...) to use GroupBy, but you can also do:

    ::

        grouped = groupby(obj, ...)

    Parameters
    ----------
    obj : pandas object
    axis : int, default 0
    level : int, default None
        Level of MultiIndex
    groupings : list of Grouping objects
        Most users should ignore this
    exclusions : array-like, optional
        List of columns to exclude
    name : str
        Most users should ignore this

    Returns
    -------
    **Attributes**
    groups : dict
        {group name -> group labels}
    len(grouped) : int
        Number of groups

    Notes
    -----
    After grouping, see aggregate, apply, and transform functions. Here are
    some other brief notes about usage. When grouping by multiple groups, the
    result index will be a MultiIndex (hierarchical) by default.

    Iteration produces (key, group) tuples, i.e. chunking the data by group. So
    you can write code like:

    ::

        grouped = obj.groupby(keys, axis=axis)
        for key, group in grouped:
            # do something with the data

    Function calls on GroupBy, if not specially implemented, "dispatch" to the
    grouped data. So if you group a DataFrame and wish to invoke the std()
    method on each group, you can simply do:

    ::

        df.groupby(mapper).std()

    rather than

    ::

        df.groupby(mapper).aggregate(np.std)

    You can pass arguments to these "wrapped" functions, too.

    See the online documentation for full exposition on these topics and much
    more
    """

    grouper: ops.BaseGrouper
    as_index: bool
    @final
    def __init__(
        self,
        obj: FrameOrSeries,
        keys: _KeysArgType | None = ...,
        axis: int = ...,
        level: IndexLabel | None = ...,
        grouper: ops.BaseGrouper | None = ...,
        exclusions: frozenset[Hashable] | None = ...,
        selection: IndexLabel | None = ...,
        as_index: bool = ...,
        sort: bool = ...,
        group_keys: bool = ...,
        squeeze: bool = ...,
        observed: bool = ...,
        mutated: bool = ...,
        dropna: bool = ...,
    ) -> None: ...
    def __getattr__(self, attr: str): ...
    @Appender(
        _apply_docs["template"].format(
            input="dataframe", examples=_apply_docs["dataframe_examples"]
        )
    )
    def apply(self, func, *args, **kwargs): ...
    @final
    @Substitution(name="groupby")
    @Appender(_common_see_also)
    def any(self, skipna: bool = ...):  # -> NoReturn:
        """
        Return True if any value in the group is truthful, else False.

        Parameters
        ----------
        skipna : bool, default True
            Flag to ignore nan values during truth testing.

        Returns
        -------
        Series or DataFrame
            DataFrame or Series of boolean values, where a value is True if any element
            is True within its respective group, False otherwise.
        """
        ...
    @final
    @Substitution(name="groupby")
    @Appender(_common_see_also)
    def all(self, skipna: bool = ...):  # -> NoReturn:
        """
        Return True if all values in the group are truthful, else False.

        Parameters
        ----------
        skipna : bool, default True
            Flag to ignore nan values during truth testing.

        Returns
        -------
        Series or DataFrame
            DataFrame or Series of boolean values, where a value is True if all elements
            are True within its respective group, False otherwise.
        """
        ...
    @Substitution(name="groupby")
    @Appender(_common_see_also)
    def count(self):
        """
        Compute count of group, excluding missing values.

        Returns
        -------
        Series or DataFrame
            Count of values within each group.
        """
        ...
    @final
    @Substitution(name="groupby")
    @Substitution(see_also=_common_see_also)
    def mean(self, numeric_only: bool | lib.NoDefault = ...):  # -> NoReturn:
        """
        Compute mean of groups, excluding missing values.

        Parameters
        ----------
        numeric_only : bool, default True
            Include only float, int, boolean columns. If None, will attempt to use
            everything, then use only numeric data.

        Returns
        -------
        pandas.Series or pandas.DataFrame
        %(see_also)s
        Examples
        --------
        >>> df = pd.DataFrame({'A': [1, 1, 2, 1, 2],
        ...                    'B': [np.nan, 2, 3, 4, 5],
        ...                    'C': [1, 2, 1, 1, 2]}, columns=['A', 'B', 'C'])

        Groupby one column and return the mean of the remaining columns in
        each group.

        >>> df.groupby('A').mean()
             B         C
        A
        1  3.0  1.333333
        2  4.0  1.500000

        Groupby two columns and return the mean of the remaining column.

        >>> df.groupby(['A', 'B']).mean()
                 C
        A B
        1 2.0  2.0
          4.0  1.0
        2 3.0  1.0
          5.0  2.0

        Groupby one column and return the mean of only particular column in
        the group.

        >>> df.groupby('A')['B'].mean()
        A
        1    3.0
        2    4.0
        Name: B, dtype: float64
        """
        ...
    @final
    @Substitution(name="groupby")
    @Appender(_common_see_also)
    def median(self, numeric_only: bool | lib.NoDefault = ...):  # -> NoReturn:
        """
        Compute median of groups, excluding missing values.

        For multiple groupings, the result index will be a MultiIndex

        Parameters
        ----------
        numeric_only : bool, default True
            Include only float, int, boolean columns. If None, will attempt to use
            everything, then use only numeric data.

        Returns
        -------
        Series or DataFrame
            Median of values within each group.
        """
        ...
    @final
    @Substitution(name="groupby")
    @Appender(_common_see_also)
    def std(self, ddof: int = ...):  # -> NoReturn:
        """
        Compute standard deviation of groups, excluding missing values.

        For multiple groupings, the result index will be a MultiIndex.

        Parameters
        ----------
        ddof : int, default 1
            Degrees of freedom.

        Returns
        -------
        Series or DataFrame
            Standard deviation of values within each group.
        """
        ...
    @final
    @Substitution(name="groupby")
    @Appender(_common_see_also)
    def var(self, ddof: int = ...):  # -> FrameOrSeriesUnion:
        """
        Compute variance of groups, excluding missing values.

        For multiple groupings, the result index will be a MultiIndex.

        Parameters
        ----------
        ddof : int, default 1
            Degrees of freedom.

        Returns
        -------
        Series or DataFrame
            Variance of values within each group.
        """
        ...
    @final
    @Substitution(name="groupby")
    @Appender(_common_see_also)
    def sem(self, ddof: int = ...):  # -> NoReturn:
        """
        Compute standard error of the mean of groups, excluding missing values.

        For multiple groupings, the result index will be a MultiIndex.

        Parameters
        ----------
        ddof : int, default 1
            Degrees of freedom.

        Returns
        -------
        Series or DataFrame
            Standard error of the mean of values within each group.
        """
        ...
    @final
    @Substitution(name="groupby")
    @Appender(_common_see_also)
    def size(self) -> FrameOrSeriesUnion:
        """
        Compute group sizes.

        Returns
        -------
        DataFrame or Series
            Number of rows in each group as a Series if as_index is True
            or a DataFrame if as_index is False.
        """
        ...
    @final
    @doc(_groupby_agg_method_template, fname="sum", no=True, mc=0)
    def sum(self, numeric_only: bool | lib.NoDefault = ..., min_count: int = ...): ...
    @final
    @doc(_groupby_agg_method_template, fname="prod", no=True, mc=0)
    def prod(self, numeric_only: bool | lib.NoDefault = ..., min_count: int = ...): ...
    @final
    @doc(_groupby_agg_method_template, fname="min", no=False, mc=-1)
    def min(self, numeric_only: bool = ..., min_count: int = ...): ...
    @final
    @doc(_groupby_agg_method_template, fname="max", no=False, mc=-1)
    def max(self, numeric_only: bool = ..., min_count: int = ...): ...
    @final
    @doc(_groupby_agg_method_template, fname="first", no=False, mc=-1)
    def first(self, numeric_only: bool = ..., min_count: int = ...): ...
    @final
    @doc(_groupby_agg_method_template, fname="last", no=False, mc=-1)
    def last(self, numeric_only: bool = ..., min_count: int = ...): ...
    @final
    @Substitution(name="groupby")
    @Appender(_common_see_also)
    def ohlc(self) -> DataFrame:
        """
        Compute open, high, low and close values of a group, excluding missing values.

        For multiple groupings, the result index will be a MultiIndex

        Returns
        -------
        DataFrame
            Open, high, low and close values within each group.
        """
        ...
    @final
    @doc(DataFrame.describe)
    def describe(self, **kwargs): ...
    @final
    def resample(self, rule, *args, **kwargs):  # -> Any:
        """
        Provide resampling when using a TimeGrouper.

        Given a grouper, the function resamples it according to a string
        "string" -> "frequency".

        See the :ref:`frequency aliases <timeseries.offset_aliases>`
        documentation for more details.

        Parameters
        ----------
        rule : str or DateOffset
            The offset string or object representing target grouper conversion.
        *args, **kwargs
            Possible arguments are `how`, `fill_method`, `limit`, `kind` and
            `on`, and other arguments of `TimeGrouper`.

        Returns
        -------
        Grouper
            Return a new grouper with our resampler appended.

        See Also
        --------
        Grouper : Specify a frequency to resample with when
            grouping by a key.
        DatetimeIndex.resample : Frequency conversion and resampling of
            time series.

        Examples
        --------
        >>> idx = pd.date_range('1/1/2000', periods=4, freq='T')
        >>> df = pd.DataFrame(data=4 * [range(2)],
        ...                   index=idx,
        ...                   columns=['a', 'b'])
        >>> df.iloc[2, 0] = 5
        >>> df
                            a  b
        2000-01-01 00:00:00  0  1
        2000-01-01 00:01:00  0  1
        2000-01-01 00:02:00  5  1
        2000-01-01 00:03:00  0  1

        Downsample the DataFrame into 3 minute bins and sum the values of
        the timestamps falling into a bin.

        >>> df.groupby('a').resample('3T').sum()
                                 a  b
        a
        0   2000-01-01 00:00:00  0  2
            2000-01-01 00:03:00  0  1
        5   2000-01-01 00:00:00  5  1

        Upsample the series into 30 second bins.

        >>> df.groupby('a').resample('30S').sum()
                            a  b
        a
        0   2000-01-01 00:00:00  0  1
            2000-01-01 00:00:30  0  0
            2000-01-01 00:01:00  0  1
            2000-01-01 00:01:30  0  0
            2000-01-01 00:02:00  0  0
            2000-01-01 00:02:30  0  0
            2000-01-01 00:03:00  0  1
        5   2000-01-01 00:02:00  5  1

        Resample by month. Values are assigned to the month of the period.

        >>> df.groupby('a').resample('M').sum()
                    a  b
        a
        0   2000-01-31  0  3
        5   2000-01-31  5  1

        Downsample the series into 3 minute bins as above, but close the right
        side of the bin interval.

        >>> df.groupby('a').resample('3T', closed='right').sum()
                                 a  b
        a
        0   1999-12-31 23:57:00  0  1
            2000-01-01 00:00:00  0  2
        5   2000-01-01 00:00:00  5  1

        Downsample the series into 3 minute bins and close the right side of
        the bin interval, but label each bin using the right edge instead of
        the left.

        >>> df.groupby('a').resample('3T', closed='right', label='right').sum()
                                 a  b
        a
        0   2000-01-01 00:00:00  0  1
            2000-01-01 00:03:00  0  2
        5   2000-01-01 00:03:00  5  1
        """
        ...
    @final
    @Substitution(name="groupby")
    @Appender(_common_see_also)
    def rolling(self, *args, **kwargs):  # -> RollingGroupby:
        """
        Return a rolling grouper, providing rolling functionality per group.
        """
        ...
    @final
    @Substitution(name="groupby")
    @Appender(_common_see_also)
    def expanding(self, *args, **kwargs):  # -> ExpandingGroupby:
        """
        Return an expanding grouper, providing expanding
        functionality per group.
        """
        ...
    @final
    @Substitution(name="groupby")
    @Appender(_common_see_also)
    def ewm(self, *args, **kwargs):  # -> ExponentialMovingWindowGroupby:
        """
        Return an ewm grouper, providing ewm functionality per group.
        """
        ...
    @final
    @Substitution(name="groupby")
    def pad(self, limit=...):  # -> NoReturn:
        """
        Forward fill the values.

        Parameters
        ----------
        limit : int, optional
            Limit of how many values to fill.

        Returns
        -------
        Series or DataFrame
            Object with missing values filled.

        See Also
        --------
        Series.pad: Returns Series with minimum number of char in object.
        DataFrame.pad: Object with missing values filled or None if inplace=True.
        Series.fillna: Fill NaN values of a Series.
        DataFrame.fillna: Fill NaN values of a DataFrame.
        """
        ...
    ffill = ...
    @final
    @Substitution(name="groupby")
    def backfill(self, limit=...):  # -> NoReturn:
        """
        Backward fill the values.

        Parameters
        ----------
        limit : int, optional
            Limit of how many values to fill.

        Returns
        -------
        Series or DataFrame
            Object with missing values filled.

        See Also
        --------
        Series.backfill :  Backward fill the missing values in the dataset.
        DataFrame.backfill:  Backward fill the missing values in the dataset.
        Series.fillna: Fill NaN values of a Series.
        DataFrame.fillna: Fill NaN values of a DataFrame.
        """
        ...
    bfill = ...
    @final
    @Substitution(name="groupby")
    @Substitution(see_also=_common_see_also)
    def nth(
        self, n: int | list[int], dropna: Literal["any", "all", None] = ...
    ) -> DataFrame:
        """
        Take the nth row from each group if n is an int, or a subset of rows
        if n is a list of ints.

        If dropna, will take the nth non-null row, dropna is either
        'all' or 'any'; this is equivalent to calling dropna(how=dropna)
        before the groupby.

        Parameters
        ----------
        n : int or list of ints
            A single nth value for the row or a list of nth values.
        dropna : {'any', 'all', None}, default None
            Apply the specified dropna operation before counting which row is
            the nth row.

        Returns
        -------
        Series or DataFrame
            N-th value within each group.
        %(see_also)s
        Examples
        --------

        >>> df = pd.DataFrame({'A': [1, 1, 2, 1, 2],
        ...                    'B': [np.nan, 2, 3, 4, 5]}, columns=['A', 'B'])
        >>> g = df.groupby('A')
        >>> g.nth(0)
             B
        A
        1  NaN
        2  3.0
        >>> g.nth(1)
             B
        A
        1  2.0
        2  5.0
        >>> g.nth(-1)
             B
        A
        1  4.0
        2  5.0
        >>> g.nth([0, 1])
             B
        A
        1  NaN
        1  2.0
        2  3.0
        2  5.0

        Specifying `dropna` allows count ignoring ``NaN``

        >>> g.nth(0, dropna='any')
             B
        A
        1  2.0
        2  3.0

        NaNs denote group exhausted when using dropna

        >>> g.nth(3, dropna='any')
            B
        A
        1 NaN
        2 NaN

        Specifying `as_index=False` in `groupby` keeps the original index.

        >>> df.groupby('A', as_index=False).nth(1)
           A    B
        1  1  2.0
        4  2  5.0
        """
        ...
    @final
    def quantile(self, q=..., interpolation: str = ...):  # -> NoReturn:
        """
        Return group values at the given quantile, a la numpy.percentile.

        Parameters
        ----------
        q : float or array-like, default 0.5 (50% quantile)
            Value(s) between 0 and 1 providing the quantile(s) to compute.
        interpolation : {'linear', 'lower', 'higher', 'midpoint', 'nearest'}
            Method to use when the desired quantile falls between two points.

        Returns
        -------
        Series or DataFrame
            Return type determined by caller of GroupBy object.

        See Also
        --------
        Series.quantile : Similar method for Series.
        DataFrame.quantile : Similar method for DataFrame.
        numpy.percentile : NumPy method to compute qth percentile.

        Examples
        --------
        >>> df = pd.DataFrame([
        ...     ['a', 1], ['a', 2], ['a', 3],
        ...     ['b', 1], ['b', 3], ['b', 5]
        ... ], columns=['key', 'val'])
        >>> df.groupby('key').quantile()
            val
        key
        a    2.0
        b    3.0
        """
        ...
    @final
    @Substitution(name="groupby")
    def ngroup(self, ascending: bool = ...):  # -> _NotImplementedType | Series:
        """
        Number each group from 0 to the number of groups - 1.

        This is the enumerative complement of cumcount.  Note that the
        numbers given to the groups match the order in which the groups
        would be seen when iterating over the groupby object, not the
        order they are first observed.

        Parameters
        ----------
        ascending : bool, default True
            If False, number in reverse, from number of group - 1 to 0.

        Returns
        -------
        Series
            Unique numbers for each group.

        See Also
        --------
        .cumcount : Number the rows in each group.

        Examples
        --------
        >>> df = pd.DataFrame({"A": list("aaabba")})
        >>> df
           A
        0  a
        1  a
        2  a
        3  b
        4  b
        5  a
        >>> df.groupby('A').ngroup()
        0    0
        1    0
        2    0
        3    1
        4    1
        5    0
        dtype: int64
        >>> df.groupby('A').ngroup(ascending=False)
        0    1
        1    1
        2    1
        3    0
        4    0
        5    1
        dtype: int64
        >>> df.groupby(["A", [1,1,2,3,2,1]]).ngroup()
        0    0
        1    0
        2    1
        3    3
        4    2
        5    0
        dtype: int64
        """
        ...
    @final
    @Substitution(name="groupby")
    def cumcount(self, ascending: bool = ...):  # -> Series:
        """
        Number each item in each group from 0 to the length of that group - 1.

        Essentially this is equivalent to

        .. code-block:: python

            self.apply(lambda x: pd.Series(np.arange(len(x)), x.index))

        Parameters
        ----------
        ascending : bool, default True
            If False, number in reverse, from length of group - 1 to 0.

        Returns
        -------
        Series
            Sequence number of each element within each group.

        See Also
        --------
        .ngroup : Number the groups themselves.

        Examples
        --------
        >>> df = pd.DataFrame([['a'], ['a'], ['a'], ['b'], ['b'], ['a']],
        ...                   columns=['A'])
        >>> df
           A
        0  a
        1  a
        2  a
        3  b
        4  b
        5  a
        >>> df.groupby('A').cumcount()
        0    0
        1    1
        2    2
        3    0
        4    1
        5    3
        dtype: int64
        >>> df.groupby('A').cumcount(ascending=False)
        0    3
        1    2
        2    1
        3    1
        4    0
        5    0
        dtype: int64
        """
        ...
    @final
    @Substitution(name="groupby")
    @Appender(_common_see_also)
    def rank(
        self,
        method: str = ...,
        ascending: bool = ...,
        na_option: str = ...,
        pct: bool = ...,
        axis: int = ...,
    ):  # -> FrameOrSeriesUnion:
        """
        Provide the rank of values within each group.

        Parameters
        ----------
        method : {'average', 'min', 'max', 'first', 'dense'}, default 'average'
            * average: average rank of group.
            * min: lowest rank in group.
            * max: highest rank in group.
            * first: ranks assigned in order they appear in the array.
            * dense: like 'min', but rank always increases by 1 between groups.
        ascending : bool, default True
            False for ranks by high (1) to low (N).
        na_option : {'keep', 'top', 'bottom'}, default 'keep'
            * keep: leave NA values where they are.
            * top: smallest rank if ascending.
            * bottom: smallest rank if descending.
        pct : bool, default False
            Compute percentage rank of data within each group.
        axis : int, default 0
            The axis of the object over which to compute the rank.

        Returns
        -------
        DataFrame with ranking of values within each group
        """
        ...
    @final
    @Substitution(name="groupby")
    @Appender(_common_see_also)
    def cumprod(self, axis=..., *args, **kwargs):  # -> FrameOrSeriesUnion:
        """
        Cumulative product for each group.

        Returns
        -------
        Series or DataFrame
        """
        ...
    @final
    @Substitution(name="groupby")
    @Appender(_common_see_also)
    def cumsum(self, axis=..., *args, **kwargs):  # -> FrameOrSeriesUnion:
        """
        Cumulative sum for each group.

        Returns
        -------
        Series or DataFrame
        """
        ...
    @final
    @Substitution(name="groupby")
    @Appender(_common_see_also)
    def cummin(self, axis=..., **kwargs):  # -> FrameOrSeriesUnion:
        """
        Cumulative min for each group.

        Returns
        -------
        Series or DataFrame
        """
        ...
    @final
    @Substitution(name="groupby")
    @Appender(_common_see_also)
    def cummax(self, axis=..., **kwargs):  # -> FrameOrSeriesUnion:
        """
        Cumulative max for each group.

        Returns
        -------
        Series or DataFrame
        """
        ...
    @final
    @Substitution(name="groupby")
    def shift(
        self, periods=..., freq=..., axis=..., fill_value=...
    ):  # -> FrameOrSeriesUnion:
        """
        Shift each group by periods observations.

        If freq is passed, the index will be increased using the periods and the freq.

        Parameters
        ----------
        periods : int, default 1
            Number of periods to shift.
        freq : str, optional
            Frequency string.
        axis : axis to shift, default 0
            Shift direction.
        fill_value : optional
            The scalar value to use for newly introduced missing values.

        Returns
        -------
        Series or DataFrame
            Object shifted within each group.

        See Also
        --------
        Index.shift : Shift values of Index.
        tshift : Shift the time index, using the index’s frequency
            if available.
        """
        ...
    @final
    @Substitution(name="groupby")
    @Appender(_common_see_also)
    def pct_change(
        self, periods=..., fill_method=..., limit=..., freq=..., axis=...
    ):  # -> FrameOrSeriesUnion | Any:
        """
        Calculate pct_change of each value to previous entry in group.

        Returns
        -------
        Series or DataFrame
            Percentage changes within each group.
        """
        ...
    @final
    @Substitution(name="groupby")
    @Substitution(see_also=_common_see_also)
    def head(self, n=...):  # -> Any:
        """
        Return first n rows of each group.

        Similar to ``.apply(lambda x: x.head(n))``, but it returns a subset of rows
        from the original DataFrame with original index and order preserved
        (``as_index`` flag is ignored).

        Does not work for negative values of `n`.

        Returns
        -------
        Series or DataFrame
        %(see_also)s
        Examples
        --------

        >>> df = pd.DataFrame([[1, 2], [1, 4], [5, 6]],
        ...                   columns=['A', 'B'])
        >>> df.groupby('A').head(1)
           A  B
        0  1  2
        2  5  6
        >>> df.groupby('A').head(-1)
        Empty DataFrame
        Columns: [A, B]
        Index: []
        """
        ...
    @final
    @Substitution(name="groupby")
    @Substitution(see_also=_common_see_also)
    def tail(self, n=...):  # -> Any:
        """
        Return last n rows of each group.

        Similar to ``.apply(lambda x: x.tail(n))``, but it returns a subset of rows
        from the original DataFrame with original index and order preserved
        (``as_index`` flag is ignored).

        Does not work for negative values of `n`.

        Returns
        -------
        Series or DataFrame
        %(see_also)s
        Examples
        --------

        >>> df = pd.DataFrame([['a', 1], ['a', 2], ['b', 1], ['b', 2]],
        ...                   columns=['A', 'B'])
        >>> df.groupby('A').tail(1)
           A  B
        1  a  2
        3  b  2
        >>> df.groupby('A').tail(-1)
        Empty DataFrame
        Columns: [A, B]
        Index: []
        """
        ...
    @final
    def sample(
        self,
        n: int | None = ...,
        frac: float | None = ...,
        replace: bool = ...,
        weights: Sequence | Series | None = ...,
        random_state=...,
    ):  # -> FrameOrSeriesUnion:
        """
        Return a random sample of items from each group.

        You can use `random_state` for reproducibility.

        .. versionadded:: 1.1.0

        Parameters
        ----------
        n : int, optional
            Number of items to return for each group. Cannot be used with
            `frac` and must be no larger than the smallest group unless
            `replace` is True. Default is one if `frac` is None.
        frac : float, optional
            Fraction of items to return. Cannot be used with `n`.
        replace : bool, default False
            Allow or disallow sampling of the same row more than once.
        weights : list-like, optional
            Default None results in equal probability weighting.
            If passed a list-like then values must have the same length as
            the underlying DataFrame or Series object and will be used as
            sampling probabilities after normalization within each group.
            Values must be non-negative with at least one positive element
            within each group.
        random_state : int, array-like, BitGenerator, np.random.RandomState, optional
            If int, array-like, or BitGenerator (NumPy>=1.17), seed for
            random number generator
            If np.random.RandomState, use as numpy RandomState object.

        Returns
        -------
        Series or DataFrame
            A new object of same type as caller containing items randomly
            sampled within each group from the caller object.

        See Also
        --------
        DataFrame.sample: Generate random samples from a DataFrame object.
        numpy.random.choice: Generate a random sample from a given 1-D numpy
            array.

        Examples
        --------
        >>> df = pd.DataFrame(
        ...     {"a": ["red"] * 2 + ["blue"] * 2 + ["black"] * 2, "b": range(6)}
        ... )
        >>> df
               a  b
        0    red  0
        1    red  1
        2   blue  2
        3   blue  3
        4  black  4
        5  black  5

        Select one row at random for each distinct value in column a. The
        `random_state` argument can be used to guarantee reproducibility:

        >>> df.groupby("a").sample(n=1, random_state=1)
               a  b
        4  black  4
        2   blue  2
        1    red  1

        Set `frac` to sample fixed proportions rather than counts:

        >>> df.groupby("a")["b"].sample(frac=0.5, random_state=2)
        5    5
        2    2
        0    0
        Name: b, dtype: int64

        Control sample probabilities within groups by setting weights:

        >>> df.groupby("a").sample(
        ...     n=1,
        ...     weights=[1, 1, 1, 0, 0, 1],
        ...     random_state=1,
        ... )
               a  b
        5  black  5
        2   blue  2
        0    red  0
        """
        ...

@doc(GroupBy)
def get_groupby(
    obj: NDFrame,
    by: _KeysArgType | None = ...,
    axis: int = ...,
    level=...,
    grouper: ops.BaseGrouper | None = ...,
    exclusions=...,
    selection=...,
    as_index: bool = ...,
    sort: bool = ...,
    group_keys: bool = ...,
    squeeze: bool = ...,
    observed: bool = ...,
    mutated: bool = ...,
    dropna: bool = ...,
) -> GroupBy: ...
