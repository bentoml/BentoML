from typing import TYPE_CHECKING

import numpy as np
from pandas import DataFrame
from pandas._libs.tslibs import Tick
from pandas._typing import NpDtype
from pandas.core.arrays import datetimelike as dtl
from pandas.core.ops.common import unpack_zerodim_and_defer

if TYPE_CHECKING: ...

class TimedeltaArray(dtl.TimelikeOps):
    """
    Pandas ExtensionArray for timedelta data.

    .. warning::

       TimedeltaArray is currently experimental, and its API may change
       without warning. In particular, :attr:`TimedeltaArray.dtype` is
       expected to change to be an instance of an ``ExtensionDtype``
       subclass.

    Parameters
    ----------
    values : array-like
        The timedelta data.

    dtype : numpy.dtype
        Currently, only ``numpy.dtype("timedelta64[ns]")`` is accepted.
    freq : Offset, optional
    copy : bool, default False
        Whether to copy the underlying array of data.

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
    __array_priority__ = ...
    _other_ops: list[str] = ...
    _bool_ops: list[str] = ...
    _object_ops: list[str] = ...
    _field_ops: list[str] = ...
    _datetimelike_ops: list[str] = ...
    _datetimelike_methods: list[str] = ...
    @property
    def dtype(self) -> np.dtype:
        """
        The dtype for the TimedeltaArray.

        .. warning::

           A future version of pandas will change dtype to be an instance
           of a :class:`pandas.api.extensions.ExtensionDtype` subclass,
           not a ``numpy.dtype``.

        Returns
        -------
        numpy.dtype
        """
        ...
    _freq = ...
    def __init__(self, values, dtype=..., freq=..., copy: bool = ...) -> None: ...
    def astype(self, dtype, copy: bool = ...): ...
    def __iter__(self): ...
    def sum(
        self,
        *,
        axis: int | None = ...,
        dtype: NpDtype | None = ...,
        out=...,
        keepdims: bool = ...,
        initial=...,
        skipna: bool = ...,
        min_count: int = ...
    ): ...
    def std(
        self,
        *,
        axis: int | None = ...,
        dtype: NpDtype | None = ...,
        out=...,
        ddof: int = ...,
        keepdims: bool = ...,
        skipna: bool = ...
    ): ...
    @unpack_zerodim_and_defer("__mul__")
    def __mul__(self, other) -> TimedeltaArray: ...
    __rmul__ = ...
    @unpack_zerodim_and_defer("__truediv__")
    def __truediv__(self, other): ...
    @unpack_zerodim_and_defer("__rtruediv__")
    def __rtruediv__(self, other): ...
    @unpack_zerodim_and_defer("__floordiv__")
    def __floordiv__(self, other): ...
    @unpack_zerodim_and_defer("__rfloordiv__")
    def __rfloordiv__(self, other): ...
    @unpack_zerodim_and_defer("__mod__")
    def __mod__(self, other): ...
    @unpack_zerodim_and_defer("__rmod__")
    def __rmod__(self, other): ...
    @unpack_zerodim_and_defer("__divmod__")
    def __divmod__(self, other): ...
    @unpack_zerodim_and_defer("__rdivmod__")
    def __rdivmod__(self, other): ...
    def __neg__(self) -> TimedeltaArray: ...
    def __pos__(self) -> TimedeltaArray: ...
    def __abs__(self) -> TimedeltaArray: ...
    def total_seconds(self) -> np.ndarray:
        """
        Return total duration of each element expressed in seconds.

        This method is available directly on TimedeltaArray, TimedeltaIndex
        and on Series containing timedelta values under the ``.dt`` namespace.

        Returns
        -------
        seconds : [ndarray, Float64Index, Series]
            When the calling object is a TimedeltaArray, the return type
            is ndarray.  When the calling object is a TimedeltaIndex,
            the return type is a Float64Index. When the calling object
            is a Series, the return type is Series of type `float64` whose
            index is the same as the original.

        See Also
        --------
        datetime.timedelta.total_seconds : Standard library version
            of this method.
        TimedeltaIndex.components : Return a DataFrame with components of
            each Timedelta.

        Examples
        --------
        **Series**

        >>> s = pd.Series(pd.to_timedelta(np.arange(5), unit='d'))
        >>> s
        0   0 days
        1   1 days
        2   2 days
        3   3 days
        4   4 days
        dtype: timedelta64[ns]

        >>> s.dt.total_seconds()
        0         0.0
        1     86400.0
        2    172800.0
        3    259200.0
        4    345600.0
        dtype: float64

        **TimedeltaIndex**

        >>> idx = pd.to_timedelta(np.arange(5), unit='d')
        >>> idx
        TimedeltaIndex(['0 days', '1 days', '2 days', '3 days', '4 days'],
                       dtype='timedelta64[ns]', freq=None)

        >>> idx.total_seconds()
        Float64Index([0.0, 86400.0, 172800.0, 259200.00000000003, 345600.0],
                     dtype='float64')
        """
        ...
    def to_pytimedelta(self) -> np.ndarray:
        """
        Return Timedelta Array/Index as object ndarray of datetime.timedelta
        objects.

        Returns
        -------
        timedeltas : ndarray[object]
        """
        ...
    days = ...
    seconds = ...
    microseconds = ...
    nanoseconds = ...
    @property
    def components(self) -> DataFrame:
        """
        Return a dataframe of the components (days, hours, minutes,
        seconds, milliseconds, microseconds, nanoseconds) of the Timedeltas.

        Returns
        -------
        DataFrame
        """
        ...

def sequence_to_td64ns(
    data, copy: bool = ..., unit=..., errors=...
) -> tuple[np.ndarray, Tick | None]:
    """
    Parameters
    ----------
    data : list-like
    copy : bool, default False
    unit : str, optional
        The timedelta unit to treat integers as multiples of. For numeric
        data this defaults to ``'ns'``.
        Must be un-specified if the data contains a str and ``errors=="raise"``.
    errors : {"raise", "coerce", "ignore"}, default "raise"
        How to handle elements that cannot be converted to timedelta64[ns].
        See ``pandas.to_timedelta`` for details.

    Returns
    -------
    converted : numpy.ndarray
        The sequence converted to a numpy array with dtype ``timedelta64[ns]``.
    inferred_freq : Tick or None
        The inferred frequency of the sequence.

    Raises
    ------
    ValueError : Data cannot be converted to timedelta64[ns].

    Notes
    -----
    Unlike `pandas.to_timedelta`, if setting ``errors=ignore`` will not cause
    errors to be ignored; they are caught and subsequently ignored at a
    higher level.
    """
    ...

def ints_to_td64ns(data, unit=...):  # -> tuple[ndarray | Unknown, bool]:
    """
    Convert an ndarray with integer-dtype to timedelta64[ns] dtype, treating
    the integers as multiples of the given timedelta unit.

    Parameters
    ----------
    data : numpy.ndarray with integer-dtype
    unit : str, default "ns"
        The timedelta unit to treat integers as multiples of.

    Returns
    -------
    numpy.ndarray : timedelta64[ns] array converted from data
    bool : whether a copy was made
    """
    ...

def objects_to_td64ns(data, unit=..., errors=...):
    """
    Convert a object-dtyped or string-dtyped array into an
    timedelta64[ns]-dtyped array.

    Parameters
    ----------
    data : ndarray or Index
    unit : str, default "ns"
        The timedelta unit to treat integers as multiples of.
        Must not be specified if the data contains a str.
    errors : {"raise", "coerce", "ignore"}, default "raise"
        How to handle elements that cannot be converted to timedelta64[ns].
        See ``pandas.to_timedelta`` for details.

    Returns
    -------
    numpy.ndarray : timedelta64[ns] array converted from data

    Raises
    ------
    ValueError : Data cannot be converted to timedelta64[ns].

    Notes
    -----
    Unlike `pandas.to_timedelta`, if setting `errors=ignore` will not cause
    errors to be ignored; they are caught and subsequently ignored at a
    higher level.
    """
    ...
