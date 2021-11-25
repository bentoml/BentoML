from pandas._libs.tslibs.offsets import DateOffset
from pandas.util._decorators import cache_readonly

_ONE_MICRO = ...
_ONE_MILLI = ...
_ONE_SECOND = ...
_ONE_MINUTE = ...
_ONE_HOUR = ...
_ONE_DAY = ...
_offset_to_period_map = ...
_need_suffix = ...

def get_period_alias(offset_str: str) -> str | None:
    """
    Alias to closest period strings BQ->Q etc.
    """
    ...

def get_offset(name: str) -> DateOffset:
    """
    Return DateOffset object associated with rule name.

    .. deprecated:: 1.0.0

    Examples
    --------
    get_offset('EOM') --> BMonthEnd(1)
    """
    ...

def infer_freq(index, warn: bool = ...) -> str | None:
    """
    Infer the most likely frequency given the input index. If the frequency is
    uncertain, a warning will be printed.

    Parameters
    ----------
    index : DatetimeIndex or TimedeltaIndex
      If passed a Series will use the values of the series (NOT THE INDEX).
    warn : bool, default True

    Returns
    -------
    str or None
        None if no discernible frequency.

    Raises
    ------
    TypeError
        If the index is not datetime-like.
    ValueError
        If there are fewer than three values.

    Examples
    --------
    >>> idx = pd.date_range(start='2020/12/01', end='2020/12/30', periods=30)
    >>> pd.infer_freq(idx)
    'D'
    """
    ...

class _FrequencyInferer:
    """
    Not sure if I can avoid the state machine here
    """

    def __init__(self, index, warn: bool = ...) -> None: ...
    @cache_readonly
    def deltas(self): ...
    @cache_readonly
    def deltas_asi8(self): ...
    @cache_readonly
    def is_unique(self) -> bool: ...
    @cache_readonly
    def is_unique_asi8(self) -> bool: ...
    def get_freq(self) -> str | None:
        """
        Find the appropriate frequency string to describe the inferred
        frequency of self.i8values

        Returns
        -------
        str or None
        """
        ...
    @cache_readonly
    def day_deltas(self): ...
    @cache_readonly
    def hour_deltas(self): ...
    @cache_readonly
    def fields(self): ...
    @cache_readonly
    def rep_stamp(self): ...
    def month_position_check(self): ...
    @cache_readonly
    def mdiffs(self): ...
    @cache_readonly
    def ydiffs(self): ...

class _TimedeltaFrequencyInferer(_FrequencyInferer): ...

def is_subperiod(source, target) -> bool:
    """
    Returns True if downsampling is possible between source and target
    frequencies

    Parameters
    ----------
    source : str or DateOffset
        Frequency converting from
    target : str or DateOffset
        Frequency converting to

    Returns
    -------
    bool
    """
    ...

def is_superperiod(source, target) -> bool:
    """
    Returns True if upsampling is possible between source and target
    frequencies

    Parameters
    ----------
    source : str or DateOffset
        Frequency converting from
    target : str or DateOffset
        Frequency converting to

    Returns
    -------
    bool
    """
    ...
