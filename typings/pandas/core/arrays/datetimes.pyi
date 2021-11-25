from datetime import tzinfo
from typing import TYPE_CHECKING, Literal, overload

import numpy as np
from pandas import DataFrame
from pandas.core.arrays import PeriodArray, TimedeltaArray
from pandas.core.arrays import datetimelike as dtl
from pandas.core.dtypes.dtypes import DatetimeTZDtype

if TYPE_CHECKING: ...
_midnight = ...

def tz_to_dtype(tz):  # -> dtype | DatetimeTZDtype:
    """
    Return a datetime64[ns] dtype appropriate for the given timezone.

    Parameters
    ----------
    tz : tzinfo or None

    Returns
    -------
    np.dtype or Datetime64TZDType
    """
    ...

class DatetimeArray(dtl.TimelikeOps, dtl.DatelikeOps):
    """
    Pandas ExtensionArray for tz-naive or tz-aware datetime data.

    .. warning::

       DatetimeArray is currently experimental, and its API may change
       without warning. In particular, :attr:`DatetimeArray.dtype` is
       expected to change to always be an instance of an ``ExtensionDtype``
       subclass.

    Parameters
    ----------
    values : Series, Index, DatetimeArray, ndarray
        The datetime data.

        For DatetimeArray `values` (or a Series or Index boxing one),
        `dtype` and `freq` will be extracted from `values`.

    dtype : numpy.dtype or DatetimeTZDtype
        Note that the only NumPy dtype allowed is 'datetime64[ns]'.
    freq : str or Offset, optional
        The frequency.
    copy : bool, default False
        Whether to copy the underlying array of values.

    Attributes
    ----------
    None

    Methods
    -------
    None
    """

    _typ = ...
    _scalar_type = ...
    _recognized_scalars = ...
    _is_recognized_dtype = ...
    _infer_matches = ...
    _bool_ops: list[str] = ...
    _object_ops: list[str] = ...
    _field_ops: list[str] = ...
    _other_ops: list[str] = ...
    _datetimelike_ops: list[str] = ...
    _datetimelike_methods: list[str] = ...
    __array_priority__ = ...
    _dtype: np.dtype | DatetimeTZDtype
    _freq = ...
    def __init__(self, values, dtype=..., freq=..., copy: bool = ...) -> None: ...
    @property
    def dtype(self) -> np.dtype | DatetimeTZDtype:
        """
        The dtype for the DatetimeArray.

        .. warning::

           A future version of pandas will change dtype to never be a
           ``numpy.dtype``. Instead, :attr:`DatetimeArray.dtype` will
           always be an instance of an ``ExtensionDtype`` subclass.

        Returns
        -------
        numpy.dtype or DatetimeTZDtype
            If the values are tz-naive, then ``np.dtype('datetime64[ns]')``
            is returned.

            If the values are tz-aware, then the ``DatetimeTZDtype``
            is returned.
        """
        ...
    @property
    def tz(self) -> tzinfo | None:
        """
        Return timezone, if any.

        Returns
        -------
        datetime.tzinfo, pytz.tzinfo.BaseTZInfo, dateutil.tz.tz.tzfile, or None
            Returns None when the array is tz-naive.
        """
        ...
    @tz.setter
    def tz(self, value): ...
    @property
    def tzinfo(self) -> tzinfo | None:
        """
        Alias for tz attribute
        """
        ...
    @property
    def is_normalized(self) -> bool:
        """
        Returns True if all of the dates are at midnight ("no time")
        """
        ...
    def __array__(self, dtype=...) -> np.ndarray: ...
    def __iter__(
        self,
    ):  # -> Generator[DTScalarOrNaT | DatetimeLikeArrayMixin | Any, None, None]:
        """
        Return an iterator over the boxed values

        Yields
        ------
        tstamp : Timestamp
        """
        ...
    def astype(self, dtype, copy: bool = ...): ...
    def tz_convert(self, tz) -> DatetimeArray:
        """
        Convert tz-aware Datetime Array/Index from one time zone to another.

        Parameters
        ----------
        tz : str, pytz.timezone, dateutil.tz.tzfile or None
            Time zone for time. Corresponding timestamps would be converted
            to this time zone of the Datetime Array/Index. A `tz` of None will
            convert to UTC and remove the timezone information.

        Returns
        -------
        Array or Index

        Raises
        ------
        TypeError
            If Datetime Array/Index is tz-naive.

        See Also
        --------
        DatetimeIndex.tz : A timezone that has a variable offset from UTC.
        DatetimeIndex.tz_localize : Localize tz-naive DatetimeIndex to a
            given time zone, or remove timezone from a tz-aware DatetimeIndex.

        Examples
        --------
        With the `tz` parameter, we can change the DatetimeIndex
        to other time zones:

        >>> dti = pd.date_range(start='2014-08-01 09:00',
        ...                     freq='H', periods=3, tz='Europe/Berlin')

        >>> dti
        DatetimeIndex(['2014-08-01 09:00:00+02:00',
                       '2014-08-01 10:00:00+02:00',
                       '2014-08-01 11:00:00+02:00'],
                      dtype='datetime64[ns, Europe/Berlin]', freq='H')

        >>> dti.tz_convert('US/Central')
        DatetimeIndex(['2014-08-01 02:00:00-05:00',
                       '2014-08-01 03:00:00-05:00',
                       '2014-08-01 04:00:00-05:00'],
                      dtype='datetime64[ns, US/Central]', freq='H')

        With the ``tz=None``, we can remove the timezone (after converting
        to UTC if necessary):

        >>> dti = pd.date_range(start='2014-08-01 09:00', freq='H',
        ...                     periods=3, tz='Europe/Berlin')

        >>> dti
        DatetimeIndex(['2014-08-01 09:00:00+02:00',
                       '2014-08-01 10:00:00+02:00',
                       '2014-08-01 11:00:00+02:00'],
                        dtype='datetime64[ns, Europe/Berlin]', freq='H')

        >>> dti.tz_convert(None)
        DatetimeIndex(['2014-08-01 07:00:00',
                       '2014-08-01 08:00:00',
                       '2014-08-01 09:00:00'],
                        dtype='datetime64[ns]', freq='H')
        """
        ...
    @dtl.ravel_compat
    def tz_localize(self, tz, ambiguous=..., nonexistent=...) -> DatetimeArray:
        """
        Localize tz-naive Datetime Array/Index to tz-aware
        Datetime Array/Index.

        This method takes a time zone (tz) naive Datetime Array/Index object
        and makes this time zone aware. It does not move the time to another
        time zone.

        This method can also be used to do the inverse -- to create a time
        zone unaware object from an aware object. To that end, pass `tz=None`.

        Parameters
        ----------
        tz : str, pytz.timezone, dateutil.tz.tzfile or None
            Time zone to convert timestamps to. Passing ``None`` will
            remove the time zone information preserving local time.
        ambiguous : 'infer', 'NaT', bool array, default 'raise'
            When clocks moved backward due to DST, ambiguous times may arise.
            For example in Central European Time (UTC+01), when going from
            03:00 DST to 02:00 non-DST, 02:30:00 local time occurs both at
            00:30:00 UTC and at 01:30:00 UTC. In such a situation, the
            `ambiguous` parameter dictates how ambiguous times should be
            handled.

            - 'infer' will attempt to infer fall dst-transition hours based on
              order
            - bool-ndarray where True signifies a DST time, False signifies a
              non-DST time (note that this flag is only applicable for
              ambiguous times)
            - 'NaT' will return NaT where there are ambiguous times
            - 'raise' will raise an AmbiguousTimeError if there are ambiguous
              times.

        nonexistent : 'shift_forward', 'shift_backward, 'NaT', timedelta, \
default 'raise'
            A nonexistent time does not exist in a particular timezone
            where clocks moved forward due to DST.

            - 'shift_forward' will shift the nonexistent time forward to the
              closest existing time
            - 'shift_backward' will shift the nonexistent time backward to the
              closest existing time
            - 'NaT' will return NaT where there are nonexistent times
            - timedelta objects will shift nonexistent times by the timedelta
            - 'raise' will raise an NonExistentTimeError if there are
              nonexistent times.

        Returns
        -------
        Same type as self
            Array/Index converted to the specified time zone.

        Raises
        ------
        TypeError
            If the Datetime Array/Index is tz-aware and tz is not None.

        See Also
        --------
        DatetimeIndex.tz_convert : Convert tz-aware DatetimeIndex from
            one time zone to another.

        Examples
        --------
        >>> tz_naive = pd.date_range('2018-03-01 09:00', periods=3)
        >>> tz_naive
        DatetimeIndex(['2018-03-01 09:00:00', '2018-03-02 09:00:00',
                       '2018-03-03 09:00:00'],
                      dtype='datetime64[ns]', freq='D')

        Localize DatetimeIndex in US/Eastern time zone:

        >>> tz_aware = tz_naive.tz_localize(tz='US/Eastern')
        >>> tz_aware
        DatetimeIndex(['2018-03-01 09:00:00-05:00',
                       '2018-03-02 09:00:00-05:00',
                       '2018-03-03 09:00:00-05:00'],
                      dtype='datetime64[ns, US/Eastern]', freq=None)

        With the ``tz=None``, we can remove the time zone information
        while keeping the local time (not converted to UTC):

        >>> tz_aware.tz_localize(None)
        DatetimeIndex(['2018-03-01 09:00:00', '2018-03-02 09:00:00',
                       '2018-03-03 09:00:00'],
                      dtype='datetime64[ns]', freq=None)

        Be careful with DST changes. When there is sequential data, pandas can
        infer the DST time:

        >>> s = pd.to_datetime(pd.Series(['2018-10-28 01:30:00',
        ...                               '2018-10-28 02:00:00',
        ...                               '2018-10-28 02:30:00',
        ...                               '2018-10-28 02:00:00',
        ...                               '2018-10-28 02:30:00',
        ...                               '2018-10-28 03:00:00',
        ...                               '2018-10-28 03:30:00']))
        >>> s.dt.tz_localize('CET', ambiguous='infer')
        0   2018-10-28 01:30:00+02:00
        1   2018-10-28 02:00:00+02:00
        2   2018-10-28 02:30:00+02:00
        3   2018-10-28 02:00:00+01:00
        4   2018-10-28 02:30:00+01:00
        5   2018-10-28 03:00:00+01:00
        6   2018-10-28 03:30:00+01:00
        dtype: datetime64[ns, CET]

        In some cases, inferring the DST is impossible. In such cases, you can
        pass an ndarray to the ambiguous parameter to set the DST explicitly

        >>> s = pd.to_datetime(pd.Series(['2018-10-28 01:20:00',
        ...                               '2018-10-28 02:36:00',
        ...                               '2018-10-28 03:46:00']))
        >>> s.dt.tz_localize('CET', ambiguous=np.array([True, True, False]))
        0   2018-10-28 01:20:00+02:00
        1   2018-10-28 02:36:00+02:00
        2   2018-10-28 03:46:00+01:00
        dtype: datetime64[ns, CET]

        If the DST transition causes nonexistent times, you can shift these
        dates forward or backwards with a timedelta object or `'shift_forward'`
        or `'shift_backwards'`.

        >>> s = pd.to_datetime(pd.Series(['2015-03-29 02:30:00',
        ...                               '2015-03-29 03:30:00']))
        >>> s.dt.tz_localize('Europe/Warsaw', nonexistent='shift_forward')
        0   2015-03-29 03:00:00+02:00
        1   2015-03-29 03:30:00+02:00
        dtype: datetime64[ns, Europe/Warsaw]

        >>> s.dt.tz_localize('Europe/Warsaw', nonexistent='shift_backward')
        0   2015-03-29 01:59:59.999999999+01:00
        1   2015-03-29 03:30:00+02:00
        dtype: datetime64[ns, Europe/Warsaw]

        >>> s.dt.tz_localize('Europe/Warsaw', nonexistent=pd.Timedelta('1H'))
        0   2015-03-29 03:30:00+02:00
        1   2015-03-29 03:30:00+02:00
        dtype: datetime64[ns, Europe/Warsaw]
        """
        ...
    def to_pydatetime(self) -> np.ndarray:
        """
        Return Datetime Array/Index as object ndarray of datetime.datetime
        objects.

        Returns
        -------
        datetimes : ndarray[object]
        """
        ...
    def normalize(self) -> DatetimeArray:
        """
        Convert times to midnight.

        The time component of the date-time is converted to midnight i.e.
        00:00:00. This is useful in cases, when the time does not matter.
        Length is unaltered. The timezones are unaffected.

        This method is available on Series with datetime values under
        the ``.dt`` accessor, and directly on Datetime Array/Index.

        Returns
        -------
        DatetimeArray, DatetimeIndex or Series
            The same type as the original data. Series will have the same
            name and index. DatetimeIndex will have the same name.

        See Also
        --------
        floor : Floor the datetimes to the specified freq.
        ceil : Ceil the datetimes to the specified freq.
        round : Round the datetimes to the specified freq.

        Examples
        --------
        >>> idx = pd.date_range(start='2014-08-01 10:00', freq='H',
        ...                     periods=3, tz='Asia/Calcutta')
        >>> idx
        DatetimeIndex(['2014-08-01 10:00:00+05:30',
                       '2014-08-01 11:00:00+05:30',
                       '2014-08-01 12:00:00+05:30'],
                        dtype='datetime64[ns, Asia/Calcutta]', freq='H')
        >>> idx.normalize()
        DatetimeIndex(['2014-08-01 00:00:00+05:30',
                       '2014-08-01 00:00:00+05:30',
                       '2014-08-01 00:00:00+05:30'],
                       dtype='datetime64[ns, Asia/Calcutta]', freq=None)
        """
        ...
    @dtl.ravel_compat
    def to_period(self, freq=...) -> PeriodArray:
        """
        Cast to PeriodArray/Index at a particular frequency.

        Converts DatetimeArray/Index to PeriodArray/Index.

        Parameters
        ----------
        freq : str or Offset, optional
            One of pandas' :ref:`offset strings <timeseries.offset_aliases>`
            or an Offset object. Will be inferred by default.

        Returns
        -------
        PeriodArray/Index

        Raises
        ------
        ValueError
            When converting a DatetimeArray/Index with non-regular values,
            so that a frequency cannot be inferred.

        See Also
        --------
        PeriodIndex: Immutable ndarray holding ordinal values.
        DatetimeIndex.to_pydatetime: Return DatetimeIndex as object.

        Examples
        --------
        >>> df = pd.DataFrame({"y": [1, 2, 3]},
        ...                   index=pd.to_datetime(["2000-03-31 00:00:00",
        ...                                         "2000-05-31 00:00:00",
        ...                                         "2000-08-31 00:00:00"]))
        >>> df.index.to_period("M")
        PeriodIndex(['2000-03', '2000-05', '2000-08'],
                    dtype='period[M]')

        Infer the daily frequency

        >>> idx = pd.date_range("2017-01-01", periods=2)
        >>> idx.to_period()
        PeriodIndex(['2017-01-01', '2017-01-02'],
                    dtype='period[D]')
        """
        ...
    def to_perioddelta(self, freq) -> TimedeltaArray:
        """
        Calculate TimedeltaArray of difference between index
        values and index converted to PeriodArray at specified
        freq. Used for vectorized offsets.

        Parameters
        ----------
        freq : Period frequency

        Returns
        -------
        TimedeltaArray/Index
        """
        ...
    def month_name(self, locale=...):  # -> ndarray:
        """
        Return the month names of the DateTimeIndex with specified locale.

        Parameters
        ----------
        locale : str, optional
            Locale determining the language in which to return the month name.
            Default is English locale.

        Returns
        -------
        Index
            Index of month names.

        Examples
        --------
        >>> idx = pd.date_range(start='2018-01', freq='M', periods=3)
        >>> idx
        DatetimeIndex(['2018-01-31', '2018-02-28', '2018-03-31'],
                      dtype='datetime64[ns]', freq='M')
        >>> idx.month_name()
        Index(['January', 'February', 'March'], dtype='object')
        """
        ...
    def day_name(self, locale=...):  # -> ndarray:
        """
        Return the day names of the DateTimeIndex with specified locale.

        Parameters
        ----------
        locale : str, optional
            Locale determining the language in which to return the day name.
            Default is English locale.

        Returns
        -------
        Index
            Index of day names.

        Examples
        --------
        >>> idx = pd.date_range(start='2018-01-01', freq='D', periods=3)
        >>> idx
        DatetimeIndex(['2018-01-01', '2018-01-02', '2018-01-03'],
                      dtype='datetime64[ns]', freq='D')
        >>> idx.day_name()
        Index(['Monday', 'Tuesday', 'Wednesday'], dtype='object')
        """
        ...
    @property
    def time(self) -> np.ndarray:
        """
        Returns numpy array of datetime.time. The time part of the Timestamps.
        """
        ...
    @property
    def timetz(self) -> np.ndarray:
        """
        Returns numpy array of datetime.time also containing timezone
        information. The time part of the Timestamps.
        """
        ...
    @property
    def date(self) -> np.ndarray:
        """
        Returns numpy array of python datetime.date objects (namely, the date
        part of Timestamps without timezone information).
        """
        ...
    def isocalendar(self) -> DataFrame:
        """
        Returns a DataFrame with the year, week, and day calculated according to
        the ISO 8601 standard.

        .. versionadded:: 1.1.0

        Returns
        -------
        DataFrame
            with columns year, week and day

        See Also
        --------
        Timestamp.isocalendar : Function return a 3-tuple containing ISO year,
            week number, and weekday for the given Timestamp object.
        datetime.date.isocalendar : Return a named tuple object with
            three components: year, week and weekday.

        Examples
        --------
        >>> idx = pd.date_range(start='2019-12-29', freq='D', periods=4)
        >>> idx.isocalendar()
                    year  week  day
        2019-12-29  2019    52    7
        2019-12-30  2020     1    1
        2019-12-31  2020     1    2
        2020-01-01  2020     1    3
        >>> idx.isocalendar().week
        2019-12-29    52
        2019-12-30     1
        2019-12-31     1
        2020-01-01     1
        Freq: D, Name: week, dtype: UInt32
        """
        ...
    @property
    def weekofyear(self):  # -> Any:
        """
        The week ordinal of the year.

        .. deprecated:: 1.1.0

        weekofyear and week have been deprecated.
        Please use DatetimeIndex.isocalendar().week instead.
        """
        ...
    week = ...
    year = ...
    month = ...
    day = ...
    hour = ...
    minute = ...
    second = ...
    microsecond = ...
    nanosecond = ...
    _dayofweek_doc = ...
    day_of_week = ...
    dayofweek = ...
    weekday = ...
    day_of_year = ...
    dayofyear = ...
    quarter = ...
    days_in_month = ...
    daysinmonth = ...
    _is_month_doc = ...
    is_month_start = ...
    is_month_end = ...
    is_quarter_start = ...
    is_quarter_end = ...
    is_year_start = ...
    is_year_end = ...
    is_leap_year = ...
    def to_julian_date(self) -> np.ndarray:
        """
        Convert Datetime Array to float64 ndarray of Julian Dates.
        0 Julian date is noon January 1, 4713 BC.
        https://en.wikipedia.org/wiki/Julian_day
        """
        ...
    def std(
        self,
        axis=...,
        dtype=...,
        out=...,
        ddof: int = ...,
        keepdims: bool = ...,
        skipna: bool = ...,
    ): ...

@overload
def sequence_to_datetimes(
    data, allow_object: Literal[False] = ..., require_iso8601: bool = ...
) -> DatetimeArray: ...
@overload
def sequence_to_datetimes(
    data, allow_object: Literal[True] = ..., require_iso8601: bool = ...
) -> np.ndarray | DatetimeArray: ...
def sequence_to_datetimes(
    data, allow_object: bool = ..., require_iso8601: bool = ...
) -> np.ndarray | DatetimeArray:
    """
    Parse/convert the passed data to either DatetimeArray or np.ndarray[object].
    """
    ...

def sequence_to_dt64ns(
    data,
    dtype=...,
    copy=...,
    tz=...,
    dayfirst=...,
    yearfirst=...,
    ambiguous=...,
    *,
    allow_object: bool = ...,
    allow_mixed: bool = ...,
    require_iso8601: bool = ...
):  # -> tuple[Unknown | ndarray, None, None] | tuple[ndarray | Unknown | Any, tzinfo | None, Unknown | Any | None]:
    """
    Parameters
    ----------
    data : list-like
    dtype : dtype, str, or None, default None
    copy : bool, default False
    tz : tzinfo, str, or None, default None
    dayfirst : bool, default False
    yearfirst : bool, default False
    ambiguous : str, bool, or arraylike, default 'raise'
        See pandas._libs.tslibs.tzconversion.tz_localize_to_utc.
    allow_object : bool, default False
        Whether to return an object-dtype ndarray instead of raising if the
        data contains more than one timezone.
    allow_mixed : bool, default False
        Interpret integers as timestamps when datetime objects are also present.
    require_iso8601 : bool, default False
        Only consider ISO-8601 formats when parsing strings.

    Returns
    -------
    result : numpy.ndarray
        The sequence converted to a numpy array with dtype ``datetime64[ns]``.
    tz : tzinfo or None
        Either the user-provided tzinfo or one inferred from the data.
    inferred_freq : Tick or None
        The inferred frequency of the sequence.

    Raises
    ------
    TypeError : PeriodDType data is passed
    """
    ...

def objects_to_datetime64ns(
    data: np.ndarray,
    dayfirst,
    yearfirst,
    utc=...,
    errors=...,
    require_iso8601: bool = ...,
    allow_object: bool = ...,
    allow_mixed: bool = ...,
):  # -> tuple[Unknown, tzinfo | None] | tuple[Unknown, tzinfo] | tuple[ndarray, None]:
    """
    Convert data to array of timestamps.

    Parameters
    ----------
    data : np.ndarray[object]
    dayfirst : bool
    yearfirst : bool
    utc : bool, default False
        Whether to convert timezone-aware timestamps to UTC.
    errors : {'raise', 'ignore', 'coerce'}
    require_iso8601 : bool, default False
    allow_object : bool
        Whether to return an object-dtype ndarray instead of raising if the
        data contains more than one timezone.
    allow_mixed : bool, default False
        Interpret integers as timestamps when datetime objects are also present.

    Returns
    -------
    result : ndarray
        np.int64 dtype if returned values represent UTC timestamps
        np.datetime64[ns] if returned values represent wall times
        object if mixed timezones
    inferred_tz : tzinfo or None

    Raises
    ------
    ValueError : if data cannot be converted to datetimes
    """
    ...

def maybe_convert_dtype(
    data, copy: bool
):  # -> tuple[Unknown, bool] | tuple[Unknown | ndarray, bool]:
    """
    Convert data based on dtype conventions, issuing deprecation warnings
    or errors where appropriate.

    Parameters
    ----------
    data : np.ndarray or pd.Index
    copy : bool

    Returns
    -------
    data : np.ndarray or pd.Index
    copy : bool

    Raises
    ------
    TypeError : PeriodDType data is passed
    """
    ...

def validate_tz_from_dtype(dtype, tz: tzinfo | None) -> tzinfo | None:
    """
    If the given dtype is a DatetimeTZDtype, extract the implied
    tzinfo object from it and check that it does not conflict with the given
    tz.

    Parameters
    ----------
    dtype : dtype, str
    tz : None, tzinfo

    Returns
    -------
    tz : consensus tzinfo

    Raises
    ------
    ValueError : on tzinfo mismatch
    """
    ...

def generate_range(start=..., end=..., periods=..., offset=...):
    """
    Generates a sequence of dates corresponding to the specified time
    offset. Similar to dateutil.rrule except uses pandas DateOffset
    objects to represent time increments.

    Parameters
    ----------
    start : datetime, (default None)
    end : datetime, (default None)
    periods : int, (default None)
    offset : DateOffset, (default BDay())

    Notes
    -----
    * This method is faster for generating weekdays than dateutil.rrule
    * At least two of (start, end, periods) must be specified.
    * If both start and end are specified, the returned dates will
    satisfy start <= date <= end.

    Returns
    -------
    dates : generator object
    """
    ...
