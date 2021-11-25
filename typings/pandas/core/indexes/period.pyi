from typing import Hashable

import numpy as np
from pandas._libs import index as libindex
from pandas._libs.tslibs import BaseOffset
from pandas._typing import Dtype
from pandas.core.arrays.period import PeriodArray
from pandas.core.indexes.datetimelike import DatetimeIndexOpsMixin
from pandas.core.indexes.datetimes import DatetimeIndex, Index
from pandas.core.indexes.extension import inherit_names
from pandas.core.indexes.numeric import Int64Index
from pandas.util._decorators import doc

_index_doc_kwargs = ...
_shared_doc_kwargs = ...

@inherit_names(
    ["strftime", "start_time", "end_time"] + PeriodArray._field_ops,
    PeriodArray,
    wrap=True,
)
@inherit_names(["is_leap_year", "_format_native_types"], PeriodArray)
class PeriodIndex(DatetimeIndexOpsMixin):
    """
    Immutable ndarray holding ordinal values indicating regular periods in time.

    Index keys are boxed to Period objects which carries the metadata (eg,
    frequency information).

    Parameters
    ----------
    data : array-like (1d int np.ndarray or PeriodArray), optional
        Optional period-like data to construct index with.
    copy : bool
        Make a copy of input ndarray.
    freq : str or period object, optional
        One of pandas period strings or corresponding objects.
    year : int, array, or Series, default None
    month : int, array, or Series, default None
    quarter : int, array, or Series, default None
    day : int, array, or Series, default None
    hour : int, array, or Series, default None
    minute : int, array, or Series, default None
    second : int, array, or Series, default None
    dtype : str or PeriodDtype, default None

    Attributes
    ----------
    day
    dayofweek
    day_of_week
    dayofyear
    day_of_year
    days_in_month
    daysinmonth
    end_time
    freq
    freqstr
    hour
    is_leap_year
    minute
    month
    quarter
    qyear
    second
    start_time
    week
    weekday
    weekofyear
    year

    Methods
    -------
    asfreq
    strftime
    to_timestamp

    See Also
    --------
    Index : The base pandas Index type.
    Period : Represents a period of time.
    DatetimeIndex : Index with datetime64 data.
    TimedeltaIndex : Index of timedelta64 data.
    period_range : Create a fixed-frequency PeriodIndex.

    Examples
    --------
    >>> idx = pd.PeriodIndex(year=[2000, 2002], quarter=[1, 3])
    >>> idx
    PeriodIndex(['2000Q1', '2002Q3'], dtype='period[Q-DEC]')
    """

    _typ = ...
    _attributes = ...
    _data: PeriodArray
    freq: BaseOffset
    _data_cls = PeriodArray
    _engine_type = libindex.PeriodEngine
    _supports_partial_string_indexing = ...
    @doc(
        PeriodArray.asfreq,
        other="pandas.arrays.PeriodArray",
        other_name="PeriodArray",
        **_shared_doc_kwargs
    )
    def asfreq(self, freq=..., how: str = ...) -> PeriodIndex: ...
    @doc(PeriodArray.to_timestamp)
    def to_timestamp(self, freq=..., how=...) -> DatetimeIndex: ...
    @property
    @doc(PeriodArray.hour.fget)
    def hour(self) -> Int64Index: ...
    @property
    @doc(PeriodArray.minute.fget)
    def minute(self) -> Int64Index: ...
    @property
    @doc(PeriodArray.second.fget)
    def second(self) -> Int64Index: ...
    def __new__(
        cls,
        data=...,
        ordinal=...,
        freq=...,
        dtype: Dtype | None = ...,
        copy: bool = ...,
        name: Hashable = ...,
        **fields
    ) -> PeriodIndex: ...
    @property
    def values(self) -> np.ndarray: ...
    def asof_locs(self, where: Index, mask: np.ndarray) -> np.ndarray:
        """
        where : array of timestamps
        mask : np.ndarray[bool]
            Array of booleans where data is not NA.
        """
        ...
    @doc(Index.astype)
    def astype(self, dtype, copy: bool = ..., how=...): ...
    @property
    def is_full(self) -> bool:
        """
        Returns True if this PeriodIndex is range-like in that all Periods
        between start and end are present, in order.
        """
        ...
    @property
    def inferred_type(self) -> str: ...
    def get_loc(self, key, method=..., tolerance=...):
        """
        Get integer location for requested label.

        Parameters
        ----------
        key : Period, NaT, str, or datetime
            String or datetime key must be parsable as Period.

        Returns
        -------
        loc : int or ndarray[int64]

        Raises
        ------
        KeyError
            Key is not present in the index.
        TypeError
            If key is listlike or otherwise not hashable.
        """
        ...

def period_range(
    start=..., end=..., periods: int | None = ..., freq=..., name=...
) -> PeriodIndex:
    """
    Return a fixed frequency PeriodIndex.

    The day (calendar) is the default frequency.

    Parameters
    ----------
    start : str or period-like, default None
        Left bound for generating periods.
    end : str or period-like, default None
        Right bound for generating periods.
    periods : int, default None
        Number of periods to generate.
    freq : str or DateOffset, optional
        Frequency alias. By default the freq is taken from `start` or `end`
        if those are Period objects. Otherwise, the default is ``"D"`` for
        daily frequency.
    name : str, default None
        Name of the resulting PeriodIndex.

    Returns
    -------
    PeriodIndex

    Notes
    -----
    Of the three parameters: ``start``, ``end``, and ``periods``, exactly two
    must be specified.

    To learn more about the frequency strings, please see `this link
    <https://pandas.pydata.org/pandas-docs/stable/user_guide/timeseries.html#offset-aliases>`__.

    Examples
    --------
    >>> pd.period_range(start='2017-01-01', end='2018-01-01', freq='M')
    PeriodIndex(['2017-01', '2017-02', '2017-03', '2017-04', '2017-05', '2017-06',
             '2017-07', '2017-08', '2017-09', '2017-10', '2017-11', '2017-12',
             '2018-01'],
            dtype='period[M]')

    If ``start`` or ``end`` are ``Period`` objects, they will be used as anchor
    endpoints for a ``PeriodIndex`` with frequency matching that of the
    ``period_range`` constructor.

    >>> pd.period_range(start=pd.Period('2017Q1', freq='Q'),
    ...                 end=pd.Period('2017Q2', freq='Q'), freq='M')
    PeriodIndex(['2017-03', '2017-04', '2017-05', '2017-06'],
                dtype='period[M]')
    """
    ...
