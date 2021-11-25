from typing import TYPE_CHECKING, Callable, Hashable, Literal

from pandas._typing import (
    FrameOrSeries,
    IndexLabel,
    T,
    TimedeltaConvertibleTypes,
    TimestampConvertibleTypes,
)
from pandas.core.base import PandasObject
from pandas.core.generic import NDFrame, _shared_docs
from pandas.core.groupby.groupby import BaseGroupBy, GroupBy, _pipe_template
from pandas.core.groupby.grouper import Grouper
from pandas.core.groupby.ops import BinGrouper
from pandas.util._decorators import (
    Appender,
    Substitution,
    deprecate_nonkeyword_arguments,
    doc,
)

if TYPE_CHECKING: ...
_shared_docs_kwargs: dict[str, str] = ...

class Resampler(BaseGroupBy, PandasObject):
    """
    Class for resampling datetimelike data, a groupby-like operation.
    See aggregate, transform, and apply functions on this object.

    It's easiest to use obj.resample(...) to use Resampler.

    Parameters
    ----------
    obj : Series or DataFrame
    groupby : TimeGrouper
    axis : int, default 0
    kind : str or None
        'period', 'timestamp' to override default index treatment

    Returns
    -------
    a Resampler of the appropriate type

    Notes
    -----
    After resampling, see aggregate, apply, and transform functions.
    """

    grouper: BinGrouper
    exclusions: frozenset[Hashable] = ...
    _attributes = ...
    def __init__(
        self,
        obj: FrameOrSeries,
        groupby: TimeGrouper,
        axis: int = ...,
        kind=...,
        *,
        selection=...,
        **kwargs
    ) -> None: ...
    def __str__(self) -> str:
        """
        Provide a nice str repr of our rolling object.
        """
        ...
    def __getattr__(self, attr: str): ...
    @property
    def obj(self) -> FrameOrSeries: ...
    @property
    def ax(self): ...
    @Substitution(
        klass="Resampler",
        examples="""
    >>> df = pd.DataFrame({'A': [1, 2, 3, 4]},
    ...                   index=pd.date_range('2012-08-02', periods=4))
    >>> df
                A
    2012-08-02  1
    2012-08-03  2
    2012-08-04  3
    2012-08-05  4

    To get the difference between each 2-day period's maximum and minimum
    value in one pass, you can do

    >>> df.resample('2D').pipe(lambda x: x.max() - x.min())
                A
    2012-08-02  1
    2012-08-04  1""",
    )
    @Appender(_pipe_template)
    def pipe(
        self, func: Callable[..., T] | tuple[Callable[..., T], str], *args, **kwargs
    ) -> T: ...
    _agg_see_also_doc = ...
    _agg_examples_doc = ...
    @doc(
        _shared_docs["aggregate"],
        see_also=_agg_see_also_doc,
        examples=_agg_examples_doc,
        klass="DataFrame",
        axis="",
    )
    def aggregate(self, func, *args, **kwargs): ...
    agg = ...
    apply = ...
    def transform(self, arg, *args, **kwargs):  # -> Any:
        """
        Call function producing a like-indexed Series on each group and return
        a Series with the transformed values.

        Parameters
        ----------
        arg : function
            To apply to each group. Should return a Series with the same index.

        Returns
        -------
        transformed : Series

        Examples
        --------
        >>> resampled.transform(lambda x: (x - x.mean()) / x.std())
        """
        ...
    def pad(self, limit=...):  # -> NoReturn:
        """
        Forward fill the values.

        Parameters
        ----------
        limit : int, optional
            Limit of how many values to fill.

        Returns
        -------
        An upsampled Series.

        See Also
        --------
        Series.fillna: Fill NA/NaN values using the specified method.
        DataFrame.fillna: Fill NA/NaN values using the specified method.
        """
        ...
    ffill = ...
    def nearest(self, limit=...):  # -> NoReturn:
        """
        Resample by using the nearest value.

        When resampling data, missing values may appear (e.g., when the
        resampling frequency is higher than the original frequency).
        The `nearest` method will replace ``NaN`` values that appeared in
        the resampled data with the value from the nearest member of the
        sequence, based on the index value.
        Missing values that existed in the original data will not be modified.
        If `limit` is given, fill only this many values in each direction for
        each of the original values.

        Parameters
        ----------
        limit : int, optional
            Limit of how many values to fill.

        Returns
        -------
        Series or DataFrame
            An upsampled Series or DataFrame with ``NaN`` values filled with
            their nearest value.

        See Also
        --------
        backfill : Backward fill the new missing values in the resampled data.
        pad : Forward fill ``NaN`` values.

        Examples
        --------
        >>> s = pd.Series([1, 2],
        ...               index=pd.date_range('20180101',
        ...                                   periods=2,
        ...                                   freq='1h'))
        >>> s
        2018-01-01 00:00:00    1
        2018-01-01 01:00:00    2
        Freq: H, dtype: int64

        >>> s.resample('15min').nearest()
        2018-01-01 00:00:00    1
        2018-01-01 00:15:00    1
        2018-01-01 00:30:00    2
        2018-01-01 00:45:00    2
        2018-01-01 01:00:00    2
        Freq: 15T, dtype: int64

        Limit the number of upsampled values imputed by the nearest:

        >>> s.resample('15min').nearest(limit=1)
        2018-01-01 00:00:00    1.0
        2018-01-01 00:15:00    1.0
        2018-01-01 00:30:00    NaN
        2018-01-01 00:45:00    2.0
        2018-01-01 01:00:00    2.0
        Freq: 15T, dtype: float64
        """
        ...
    def backfill(self, limit=...):  # -> NoReturn:
        """
        Backward fill the new missing values in the resampled data.

        In statistics, imputation is the process of replacing missing data with
        substituted values [1]_. When resampling data, missing values may
        appear (e.g., when the resampling frequency is higher than the original
        frequency). The backward fill will replace NaN values that appeared in
        the resampled data with the next value in the original sequence.
        Missing values that existed in the original data will not be modified.

        Parameters
        ----------
        limit : int, optional
            Limit of how many values to fill.

        Returns
        -------
        Series, DataFrame
            An upsampled Series or DataFrame with backward filled NaN values.

        See Also
        --------
        bfill : Alias of backfill.
        fillna : Fill NaN values using the specified method, which can be
            'backfill'.
        nearest : Fill NaN values with nearest neighbor starting from center.
        pad : Forward fill NaN values.
        Series.fillna : Fill NaN values in the Series using the
            specified method, which can be 'backfill'.
        DataFrame.fillna : Fill NaN values in the DataFrame using the
            specified method, which can be 'backfill'.

        References
        ----------
        .. [1] https://en.wikipedia.org/wiki/Imputation_(statistics)

        Examples
        --------
        Resampling a Series:

        >>> s = pd.Series([1, 2, 3],
        ...               index=pd.date_range('20180101', periods=3, freq='h'))
        >>> s
        2018-01-01 00:00:00    1
        2018-01-01 01:00:00    2
        2018-01-01 02:00:00    3
        Freq: H, dtype: int64

        >>> s.resample('30min').backfill()
        2018-01-01 00:00:00    1
        2018-01-01 00:30:00    2
        2018-01-01 01:00:00    2
        2018-01-01 01:30:00    3
        2018-01-01 02:00:00    3
        Freq: 30T, dtype: int64

        >>> s.resample('15min').backfill(limit=2)
        2018-01-01 00:00:00    1.0
        2018-01-01 00:15:00    NaN
        2018-01-01 00:30:00    2.0
        2018-01-01 00:45:00    2.0
        2018-01-01 01:00:00    2.0
        2018-01-01 01:15:00    NaN
        2018-01-01 01:30:00    3.0
        2018-01-01 01:45:00    3.0
        2018-01-01 02:00:00    3.0
        Freq: 15T, dtype: float64

        Resampling a DataFrame that has missing values:

        >>> df = pd.DataFrame({'a': [2, np.nan, 6], 'b': [1, 3, 5]},
        ...                   index=pd.date_range('20180101', periods=3,
        ...                                       freq='h'))
        >>> df
                               a  b
        2018-01-01 00:00:00  2.0  1
        2018-01-01 01:00:00  NaN  3
        2018-01-01 02:00:00  6.0  5

        >>> df.resample('30min').backfill()
                               a  b
        2018-01-01 00:00:00  2.0  1
        2018-01-01 00:30:00  NaN  3
        2018-01-01 01:00:00  NaN  3
        2018-01-01 01:30:00  6.0  5
        2018-01-01 02:00:00  6.0  5

        >>> df.resample('15min').backfill(limit=2)
                               a    b
        2018-01-01 00:00:00  2.0  1.0
        2018-01-01 00:15:00  NaN  NaN
        2018-01-01 00:30:00  NaN  3.0
        2018-01-01 00:45:00  NaN  3.0
        2018-01-01 01:00:00  NaN  3.0
        2018-01-01 01:15:00  NaN  NaN
        2018-01-01 01:30:00  6.0  5.0
        2018-01-01 01:45:00  6.0  5.0
        2018-01-01 02:00:00  6.0  5.0
        """
        ...
    bfill = ...
    def fillna(self, method, limit=...):  # -> NoReturn:
        """
        Fill missing values introduced by upsampling.

        In statistics, imputation is the process of replacing missing data with
        substituted values [1]_. When resampling data, missing values may
        appear (e.g., when the resampling frequency is higher than the original
        frequency).

        Missing values that existed in the original data will
        not be modified.

        Parameters
        ----------
        method : {'pad', 'backfill', 'ffill', 'bfill', 'nearest'}
            Method to use for filling holes in resampled data

            * 'pad' or 'ffill': use previous valid observation to fill gap
              (forward fill).
            * 'backfill' or 'bfill': use next valid observation to fill gap.
            * 'nearest': use nearest valid observation to fill gap.

        limit : int, optional
            Limit of how many consecutive missing values to fill.

        Returns
        -------
        Series or DataFrame
            An upsampled Series or DataFrame with missing values filled.

        See Also
        --------
        backfill : Backward fill NaN values in the resampled data.
        pad : Forward fill NaN values in the resampled data.
        nearest : Fill NaN values in the resampled data
            with nearest neighbor starting from center.
        interpolate : Fill NaN values using interpolation.
        Series.fillna : Fill NaN values in the Series using the
            specified method, which can be 'bfill' and 'ffill'.
        DataFrame.fillna : Fill NaN values in the DataFrame using the
            specified method, which can be 'bfill' and 'ffill'.

        References
        ----------
        .. [1] https://en.wikipedia.org/wiki/Imputation_(statistics)

        Examples
        --------
        Resampling a Series:

        >>> s = pd.Series([1, 2, 3],
        ...               index=pd.date_range('20180101', periods=3, freq='h'))
        >>> s
        2018-01-01 00:00:00    1
        2018-01-01 01:00:00    2
        2018-01-01 02:00:00    3
        Freq: H, dtype: int64

        Without filling the missing values you get:

        >>> s.resample("30min").asfreq()
        2018-01-01 00:00:00    1.0
        2018-01-01 00:30:00    NaN
        2018-01-01 01:00:00    2.0
        2018-01-01 01:30:00    NaN
        2018-01-01 02:00:00    3.0
        Freq: 30T, dtype: float64

        >>> s.resample('30min').fillna("backfill")
        2018-01-01 00:00:00    1
        2018-01-01 00:30:00    2
        2018-01-01 01:00:00    2
        2018-01-01 01:30:00    3
        2018-01-01 02:00:00    3
        Freq: 30T, dtype: int64

        >>> s.resample('15min').fillna("backfill", limit=2)
        2018-01-01 00:00:00    1.0
        2018-01-01 00:15:00    NaN
        2018-01-01 00:30:00    2.0
        2018-01-01 00:45:00    2.0
        2018-01-01 01:00:00    2.0
        2018-01-01 01:15:00    NaN
        2018-01-01 01:30:00    3.0
        2018-01-01 01:45:00    3.0
        2018-01-01 02:00:00    3.0
        Freq: 15T, dtype: float64

        >>> s.resample('30min').fillna("pad")
        2018-01-01 00:00:00    1
        2018-01-01 00:30:00    1
        2018-01-01 01:00:00    2
        2018-01-01 01:30:00    2
        2018-01-01 02:00:00    3
        Freq: 30T, dtype: int64

        >>> s.resample('30min').fillna("nearest")
        2018-01-01 00:00:00    1
        2018-01-01 00:30:00    2
        2018-01-01 01:00:00    2
        2018-01-01 01:30:00    3
        2018-01-01 02:00:00    3
        Freq: 30T, dtype: int64

        Missing values present before the upsampling are not affected.

        >>> sm = pd.Series([1, None, 3],
        ...               index=pd.date_range('20180101', periods=3, freq='h'))
        >>> sm
        2018-01-01 00:00:00    1.0
        2018-01-01 01:00:00    NaN
        2018-01-01 02:00:00    3.0
        Freq: H, dtype: float64

        >>> sm.resample('30min').fillna('backfill')
        2018-01-01 00:00:00    1.0
        2018-01-01 00:30:00    NaN
        2018-01-01 01:00:00    NaN
        2018-01-01 01:30:00    3.0
        2018-01-01 02:00:00    3.0
        Freq: 30T, dtype: float64

        >>> sm.resample('30min').fillna('pad')
        2018-01-01 00:00:00    1.0
        2018-01-01 00:30:00    1.0
        2018-01-01 01:00:00    NaN
        2018-01-01 01:30:00    NaN
        2018-01-01 02:00:00    3.0
        Freq: 30T, dtype: float64

        >>> sm.resample('30min').fillna('nearest')
        2018-01-01 00:00:00    1.0
        2018-01-01 00:30:00    NaN
        2018-01-01 01:00:00    NaN
        2018-01-01 01:30:00    3.0
        2018-01-01 02:00:00    3.0
        Freq: 30T, dtype: float64

        DataFrame resampling is done column-wise. All the same options are
        available.

        >>> df = pd.DataFrame({'a': [2, np.nan, 6], 'b': [1, 3, 5]},
        ...                   index=pd.date_range('20180101', periods=3,
        ...                                       freq='h'))
        >>> df
                               a  b
        2018-01-01 00:00:00  2.0  1
        2018-01-01 01:00:00  NaN  3
        2018-01-01 02:00:00  6.0  5

        >>> df.resample('30min').fillna("bfill")
                               a  b
        2018-01-01 00:00:00  2.0  1
        2018-01-01 00:30:00  NaN  3
        2018-01-01 01:00:00  NaN  3
        2018-01-01 01:30:00  6.0  5
        2018-01-01 02:00:00  6.0  5
        """
        ...
    @deprecate_nonkeyword_arguments(version=None, allowed_args=["self", "method"])
    @doc(NDFrame.interpolate, **_shared_docs_kwargs)
    def interpolate(
        self,
        method=...,
        axis=...,
        limit=...,
        inplace=...,
        limit_direction=...,
        limit_area=...,
        downcast=...,
        **kwargs
    ):  # -> NoReturn:
        """
        Interpolate values according to different methods.
        """
        ...
    def asfreq(self, fill_value=...):  # -> NoReturn:
        """
        Return the values at the new freq, essentially a reindex.

        Parameters
        ----------
        fill_value : scalar, optional
            Value to use for missing values, applied during upsampling (note
            this does not fill NaNs that already were present).

        Returns
        -------
        DataFrame or Series
            Values at the specified freq.

        See Also
        --------
        Series.asfreq: Convert TimeSeries to specified frequency.
        DataFrame.asfreq: Convert TimeSeries to specified frequency.
        """
        ...
    def std(self, ddof=..., *args, **kwargs):  # -> NoReturn:
        """
        Compute standard deviation of groups, excluding missing values.

        Parameters
        ----------
        ddof : int, default 1
            Degrees of freedom.

        Returns
        -------
        DataFrame or Series
            Standard deviation of values within each group.
        """
        ...
    def var(self, ddof=..., *args, **kwargs):  # -> NoReturn:
        """
        Compute variance of groups, excluding missing values.

        Parameters
        ----------
        ddof : int, default 1
            Degrees of freedom.

        Returns
        -------
        DataFrame or Series
            Variance of values within each group.
        """
        ...
    @doc(GroupBy.size)
    def size(self): ...
    @doc(GroupBy.count)
    def count(self): ...
    def quantile(self, q=..., **kwargs):  # -> NoReturn:
        """
        Return value at the given quantile.

        Parameters
        ----------
        q : float or array-like, default 0.5 (50% quantile)

        Returns
        -------
        DataFrame or Series
            Quantile of values within each group.

        See Also
        --------
        Series.quantile
            Return a series, where the index is q and the values are the quantiles.
        DataFrame.quantile
            Return a DataFrame, where the columns are the columns of self,
            and the values are the quantiles.
        DataFrameGroupBy.quantile
            Return a DataFrame, where the coulmns are groupby columns,
            and the values are its quantiles.
        """
        ...

class _GroupByMixin(PandasObject):
    """
    Provide the groupby facilities.
    """

    _attributes: list[str]
    _selection: IndexLabel | None = ...
    def __init__(self, obj, parent=..., groupby=..., **kwargs) -> None: ...
    _upsample = ...
    _downsample = ...
    _groupby_and_aggregate = ...

class DatetimeIndexResampler(Resampler): ...

class DatetimeIndexResamplerGroupby(_GroupByMixin, DatetimeIndexResampler):
    """
    Provides a resample of a groupby implementation
    """

    ...

class PeriodIndexResampler(DatetimeIndexResampler): ...

class PeriodIndexResamplerGroupby(_GroupByMixin, PeriodIndexResampler):
    """
    Provides a resample of a groupby implementation.
    """

    ...

class TimedeltaIndexResampler(DatetimeIndexResampler): ...

class TimedeltaIndexResamplerGroupby(_GroupByMixin, TimedeltaIndexResampler):
    """
    Provides a resample of a groupby implementation.
    """

    ...

def get_resampler(
    obj, kind=..., **kwds
):  # -> DatetimeIndexResampler | PeriodIndexResampler | TimedeltaIndexResampler:
    """
    Create a TimeGrouper and return our resampler.
    """
    ...

def get_resampler_for_grouping(
    groupby, rule, how=..., fill_method=..., limit=..., kind=..., on=..., **kwargs
):  # -> Any:
    """
    Return our appropriate resampler when grouping as well.
    """
    ...

class TimeGrouper(Grouper):
    """
    Custom groupby class for time-interval grouping.

    Parameters
    ----------
    freq : pandas date offset or offset alias for identifying bin edges
    closed : closed end of interval; 'left' or 'right'
    label : interval boundary to use for labeling; 'left' or 'right'
    convention : {'start', 'end', 'e', 's'}
        If axis is PeriodIndex
    """

    _attributes = ...
    def __init__(
        self,
        freq=...,
        closed: Literal["left", "right"] | None = ...,
        label: str | None = ...,
        how=...,
        axis=...,
        fill_method=...,
        limit=...,
        loffset=...,
        kind: str | None = ...,
        convention: str | None = ...,
        base: int | None = ...,
        origin: str | TimestampConvertibleTypes = ...,
        offset: TimedeltaConvertibleTypes | None = ...,
        **kwargs
    ) -> None: ...

def asfreq(
    obj: FrameOrSeries, freq, method=..., how=..., normalize: bool = ..., fill_value=...
) -> FrameOrSeries:
    """
    Utility frequency conversion method for Series/DataFrame.

    See :meth:`pandas.NDFrame.asfreq` for full documentation.
    """
    ...
