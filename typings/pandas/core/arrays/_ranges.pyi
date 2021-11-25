from pandas._libs.tslibs import BaseOffset, Timedelta, Timestamp

"""
Helper functions to generate range-like data for DatetimeArray
(and possibly TimedeltaArray/PeriodArray)
"""

def generate_regular_range(
    start: Timestamp | Timedelta,
    end: Timestamp | Timedelta,
    periods: int,
    freq: BaseOffset,
):  # -> ndarray:
    """
    Generate a range of dates or timestamps with the spans between dates
    described by the given `freq` DateOffset.

    Parameters
    ----------
    start : Timedelta, Timestamp or None
        First point of produced date range.
    end : Timedelta, Timestamp or None
        Last point of produced date range.
    periods : int
        Number of periods in produced date range.
    freq : Tick
        Describes space between dates in produced date range.

    Returns
    -------
    ndarray[np.int64] Representing nanoseconds.
    """
    ...
