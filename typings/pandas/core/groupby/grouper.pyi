from typing import Hashable

import numpy as np
from pandas._typing import ArrayLike, FrameOrSeries, final
from pandas.core.arrays import Categorical
from pandas.core.groupby import ops
from pandas.core.indexes.api import Index
from pandas.util._decorators import cache_readonly

"""
Provide user facing operators for doing the split part of the
split-apply-combine paradigm.
"""

class Grouper:
    """
    A Grouper allows the user to specify a groupby instruction for an object.

    This specification will select a column via the key parameter, or if the
    level and/or axis parameters are given, a level of the index of the target
    object.

    If `axis` and/or `level` are passed as keywords to both `Grouper` and
    `groupby`, the values passed to `Grouper` take precedence.

    Parameters
    ----------
    key : str, defaults to None
        Groupby key, which selects the grouping column of the target.
    level : name/number, defaults to None
        The level for the target index.
    freq : str / frequency object, defaults to None
        This will groupby the specified frequency if the target selection
        (via key or level) is a datetime-like object. For full specification
        of available frequencies, please see `here
        <https://pandas.pydata.org/pandas-docs/stable/user_guide/timeseries.html#offset-aliases>`_.
    axis : str, int, defaults to 0
        Number/name of the axis.
    sort : bool, default to False
        Whether to sort the resulting labels.
    closed : {'left' or 'right'}
        Closed end of interval. Only when `freq` parameter is passed.
    label : {'left' or 'right'}
        Interval boundary to use for labeling.
        Only when `freq` parameter is passed.
    convention : {'start', 'end', 'e', 's'}
        If grouper is PeriodIndex and `freq` parameter is passed.
    base : int, default 0
        Only when `freq` parameter is passed.
        For frequencies that evenly subdivide 1 day, the "origin" of the
        aggregated intervals. For example, for '5min' frequency, base could
        range from 0 through 4. Defaults to 0.

        .. deprecated:: 1.1.0
            The new arguments that you should use are 'offset' or 'origin'.

    loffset : str, DateOffset, timedelta object
        Only when `freq` parameter is passed.

        .. deprecated:: 1.1.0
            loffset is only working for ``.resample(...)`` and not for
            Grouper (:issue:`28302`).
            However, loffset is also deprecated for ``.resample(...)``
            See: :class:`DataFrame.resample`

    origin : {{'epoch', 'start', 'start_day', 'end', 'end_day'}}, Timestamp
        or str, default 'start_day'
        The timestamp on which to adjust the grouping. The timezone of origin must
        match the timezone of the index.
        If a timestamp is not used, these values are also supported:

        - 'epoch': `origin` is 1970-01-01
        - 'start': `origin` is the first value of the timeseries
        - 'start_day': `origin` is the first day at midnight of the timeseries

        .. versionadded:: 1.1.0

        - 'end': `origin` is the last value of the timeseries
        - 'end_day': `origin` is the ceiling midnight of the last day

        .. versionadded:: 1.3.0

    offset : Timedelta or str, default is None
        An offset timedelta added to the origin.

        .. versionadded:: 1.1.0

    dropna : bool, default True
        If True, and if group keys contain NA values, NA values together with
        row/column will be dropped. If False, NA values will also be treated as
        the key in groups.

        .. versionadded:: 1.2.0

    Returns
    -------
    A specification for a groupby instruction

    Examples
    --------
    Syntactic sugar for ``df.groupby('A')``

    >>> df = pd.DataFrame(
    ...     {
    ...         "Animal": ["Falcon", "Parrot", "Falcon", "Falcon", "Parrot"],
    ...         "Speed": [100, 5, 200, 300, 15],
    ...     }
    ... )
    >>> df
       Animal  Speed
    0  Falcon    100
    1  Parrot      5
    2  Falcon    200
    3  Falcon    300
    4  Parrot     15
    >>> df.groupby(pd.Grouper(key="Animal")).mean()
            Speed
    Animal
    Falcon  200.0
    Parrot   10.0

    Specify a resample operation on the column 'Publish date'

    >>> df = pd.DataFrame(
    ...    {
    ...        "Publish date": [
    ...             pd.Timestamp("2000-01-02"),
    ...             pd.Timestamp("2000-01-02"),
    ...             pd.Timestamp("2000-01-09"),
    ...             pd.Timestamp("2000-01-16")
    ...         ],
    ...         "ID": [0, 1, 2, 3],
    ...         "Price": [10, 20, 30, 40]
    ...     }
    ... )
    >>> df
      Publish date  ID  Price
    0   2000-01-02   0     10
    1   2000-01-02   1     20
    2   2000-01-09   2     30
    3   2000-01-16   3     40
    >>> df.groupby(pd.Grouper(key="Publish date", freq="1W")).mean()
                   ID  Price
    Publish date
    2000-01-02    0.5   15.0
    2000-01-09    2.0   30.0
    2000-01-16    3.0   40.0

    If you want to adjust the start of the bins based on a fixed timestamp:

    >>> start, end = '2000-10-01 23:30:00', '2000-10-02 00:30:00'
    >>> rng = pd.date_range(start, end, freq='7min')
    >>> ts = pd.Series(np.arange(len(rng)) * 3, index=rng)
    >>> ts
    2000-10-01 23:30:00     0
    2000-10-01 23:37:00     3
    2000-10-01 23:44:00     6
    2000-10-01 23:51:00     9
    2000-10-01 23:58:00    12
    2000-10-02 00:05:00    15
    2000-10-02 00:12:00    18
    2000-10-02 00:19:00    21
    2000-10-02 00:26:00    24
    Freq: 7T, dtype: int64

    >>> ts.groupby(pd.Grouper(freq='17min')).sum()
    2000-10-01 23:14:00     0
    2000-10-01 23:31:00     9
    2000-10-01 23:48:00    21
    2000-10-02 00:05:00    54
    2000-10-02 00:22:00    24
    Freq: 17T, dtype: int64

    >>> ts.groupby(pd.Grouper(freq='17min', origin='epoch')).sum()
    2000-10-01 23:18:00     0
    2000-10-01 23:35:00    18
    2000-10-01 23:52:00    27
    2000-10-02 00:09:00    39
    2000-10-02 00:26:00    24
    Freq: 17T, dtype: int64

    >>> ts.groupby(pd.Grouper(freq='17min', origin='2000-01-01')).sum()
    2000-10-01 23:24:00     3
    2000-10-01 23:41:00    15
    2000-10-01 23:58:00    45
    2000-10-02 00:15:00    45
    Freq: 17T, dtype: int64

    If you want to adjust the start of the bins with an `offset` Timedelta, the two
    following lines are equivalent:

    >>> ts.groupby(pd.Grouper(freq='17min', origin='start')).sum()
    2000-10-01 23:30:00     9
    2000-10-01 23:47:00    21
    2000-10-02 00:04:00    54
    2000-10-02 00:21:00    24
    Freq: 17T, dtype: int64

    >>> ts.groupby(pd.Grouper(freq='17min', offset='23h30min')).sum()
    2000-10-01 23:30:00     9
    2000-10-01 23:47:00    21
    2000-10-02 00:04:00    54
    2000-10-02 00:21:00    24
    Freq: 17T, dtype: int64

    To replace the use of the deprecated `base` argument, you can now use `offset`,
    in this example it is equivalent to have `base=2`:

    >>> ts.groupby(pd.Grouper(freq='17min', offset='2min')).sum()
    2000-10-01 23:16:00     0
    2000-10-01 23:33:00     9
    2000-10-01 23:50:00    36
    2000-10-02 00:07:00    39
    2000-10-02 00:24:00    24
    Freq: 17T, dtype: int64
    """

    axis: int
    sort: bool
    dropna: bool
    _gpr_index: Index | None
    _grouper: Index | None
    _attributes: tuple[str, ...] = ...
    def __new__(cls, *args, **kwargs): ...
    def __init__(
        self,
        key=...,
        level=...,
        freq=...,
        axis: int = ...,
        sort: bool = ...,
        dropna: bool = ...,
    ) -> None: ...
    @final
    @property
    def ax(self) -> Index: ...
    @final
    @property
    def groups(self): ...
    @final
    def __repr__(self) -> str: ...

@final
class Grouping:
    """
    Holds the grouping information for a single key

    Parameters
    ----------
    index : Index
    grouper :
    obj : DataFrame or Series
    name : Label
    level :
    observed : bool, default False
        If we are a Categorical, use the observed values
    in_axis : if the Grouping is a column in self.obj and hence among
        Groupby.exclusions list

    Returns
    -------
    **Attributes**:
      * indices : dict of {group -> index_list}
      * codes : ndarray, group codes
      * group_index : unique groups
      * groups : dict of {group -> label_list}
    """

    _codes: np.ndarray | None = ...
    _group_index: Index | None = ...
    _passed_categorical: bool
    _all_grouper: Categorical | None
    _index: Index
    def __init__(
        self,
        index: Index,
        grouper=...,
        obj: FrameOrSeries | None = ...,
        level=...,
        sort: bool = ...,
        observed: bool = ...,
        in_axis: bool = ...,
        dropna: bool = ...,
    ) -> None: ...
    def __repr__(self) -> str: ...
    def __iter__(self): ...
    @cache_readonly
    def name(self) -> Hashable: ...
    @property
    def ngroups(self) -> int: ...
    @cache_readonly
    def indices(self): ...
    @property
    def codes(self) -> np.ndarray: ...
    @cache_readonly
    def group_arraylike(self) -> ArrayLike:
        """
        Analogous to result_index, but holding an ArrayLike to ensure
        we can can retain ExtensionDtypes.
        """
        ...
    @cache_readonly
    def result_index(self) -> Index: ...
    @cache_readonly
    def group_index(self) -> Index: ...
    @cache_readonly
    def groups(self) -> dict[Hashable, np.ndarray]: ...

def get_grouper(
    obj: FrameOrSeries,
    key=...,
    axis: int = ...,
    level=...,
    sort: bool = ...,
    observed: bool = ...,
    mutated: bool = ...,
    validate: bool = ...,
    dropna: bool = ...,
) -> tuple[ops.BaseGrouper, frozenset[Hashable], FrameOrSeries]:
    """
    Create and return a BaseGrouper, which is an internal
    mapping of how to create the grouper indexers.
    This may be composed of multiple Grouping objects, indicating
    multiple groupers

    Groupers are ultimately index mappings. They can originate as:
    index mappings, keys to columns, functions, or Groupers

    Groupers enable local references to axis,level,sort, while
    the passed in axis, level, and sort are 'global'.

    This routine tries to figure out what the passing in references
    are and then creates a Grouping for each one, combined into
    a BaseGrouper.

    If observed & we have a categorical grouper, only show the observed
    values.

    If validate, then check for key/level overlaps.

    """
    ...
