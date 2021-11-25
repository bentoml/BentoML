from typing import Any

import numpy as np
from pandas._typing import ArrayLike, F
from pandas.core.dtypes.dtypes import PeriodDtype

bn = ...
_BOTTLENECK_INSTALLED = ...
_USE_BOTTLENECK = ...

def set_use_bottleneck(v: bool = ...) -> None: ...

class disallow:
    def __init__(self, *dtypes) -> None: ...
    def check(self, obj) -> bool: ...
    def __call__(self, f: F) -> F: ...

class bottleneck_switch:
    def __init__(self, name=..., **kwargs) -> None: ...
    def __call__(self, alt: F) -> F: ...

def nanany(
    values: np.ndarray,
    *,
    axis: int | None = ...,
    skipna: bool = ...,
    mask: np.ndarray | None = ...,
) -> bool:
    """
    Check if any elements along an axis evaluate to True.

    Parameters
    ----------
    values : ndarray
    axis : int, optional
    skipna : bool, default True
    mask : ndarray[bool], optional
        nan-mask if known

    Returns
    -------
    result : bool

    Examples
    --------
    >>> import pandas.core.nanops as nanops
    >>> s = pd.Series([1, 2])
    >>> nanops.nanany(s)
    True

    >>> import pandas.core.nanops as nanops
    >>> s = pd.Series([np.nan])
    >>> nanops.nanany(s)
    False
    """
    ...

def nanall(
    values: np.ndarray,
    *,
    axis: int | None = ...,
    skipna: bool = ...,
    mask: np.ndarray | None = ...,
) -> bool:
    """
    Check if all elements along an axis evaluate to True.

    Parameters
    ----------
    values : ndarray
    axis : int, optional
    skipna : bool, default True
    mask : ndarray[bool], optional
        nan-mask if known

    Returns
    -------
    result : bool

    Examples
    --------
    >>> import pandas.core.nanops as nanops
    >>> s = pd.Series([1, 2, np.nan])
    >>> nanops.nanall(s)
    True

    >>> import pandas.core.nanops as nanops
    >>> s = pd.Series([1, 0])
    >>> nanops.nanall(s)
    False
    """
    ...

@disallow("M8")
@_datetimelike_compat
def nansum(
    values: np.ndarray,
    *,
    axis: int | None = ...,
    skipna: bool = ...,
    min_count: int = ...,
    mask: np.ndarray | None = ...,
) -> float:
    """
    Sum the elements along an axis ignoring NaNs

    Parameters
    ----------
    values : ndarray[dtype]
    axis : int, optional
    skipna : bool, default True
    min_count: int, default 0
    mask : ndarray[bool], optional
        nan-mask if known

    Returns
    -------
    result : dtype

    Examples
    --------
    >>> import pandas.core.nanops as nanops
    >>> s = pd.Series([1, 2, np.nan])
    >>> nanops.nansum(s)
    3.0
    """
    ...

@disallow(PeriodDtype)
@bottleneck_switch()
@_datetimelike_compat
def nanmean(
    values: np.ndarray,
    *,
    axis: int | None = ...,
    skipna: bool = ...,
    mask: np.ndarray | None = ...,
) -> float:
    """
    Compute the mean of the element along an axis ignoring NaNs

    Parameters
    ----------
    values : ndarray
    axis : int, optional
    skipna : bool, default True
    mask : ndarray[bool], optional
        nan-mask if known

    Returns
    -------
    float
        Unless input is a float array, in which case use the same
        precision as the input array.

    Examples
    --------
    >>> import pandas.core.nanops as nanops
    >>> s = pd.Series([1, 2, np.nan])
    >>> nanops.nanmean(s)
    1.5
    """
    ...

@bottleneck_switch()
def nanmedian(values, *, axis=..., skipna=..., mask=...):
    """
    Parameters
    ----------
    values : ndarray
    axis : int, optional
    skipna : bool, default True
    mask : ndarray[bool], optional
        nan-mask if known

    Returns
    -------
    result : float
        Unless input is a float array, in which case use the same
        precision as the input array.

    Examples
    --------
    >>> import pandas.core.nanops as nanops
    >>> s = pd.Series([1, np.nan, 2, 2])
    >>> nanops.nanmedian(s)
    2.0
    """
    ...

def get_empty_reduction_result(
    shape: tuple[int, ...],
    axis: int,
    dtype: np.dtype | type[np.floating],
    fill_value: Any,
) -> np.ndarray:
    """
    The result from a reduction on an empty ndarray.

    Parameters
    ----------
    shape : Tuple[int]
    axis : int
    dtype : np.dtype
    fill_value : Any

    Returns
    -------
    np.ndarray
    """
    ...

@bottleneck_switch(ddof=1)
def nanstd(
    values, *, axis=..., skipna=..., ddof=..., mask=...
):  # -> Any | datetime64 | NDArray | Timedelta | NaTType:
    """
    Compute the standard deviation along given axis while ignoring NaNs

    Parameters
    ----------
    values : ndarray
    axis : int, optional
    skipna : bool, default True
    ddof : int, default 1
        Delta Degrees of Freedom. The divisor used in calculations is N - ddof,
        where N represents the number of elements.
    mask : ndarray[bool], optional
        nan-mask if known

    Returns
    -------
    result : float
        Unless input is a float array, in which case use the same
        precision as the input array.

    Examples
    --------
    >>> import pandas.core.nanops as nanops
    >>> s = pd.Series([1, np.nan, 2, 3])
    >>> nanops.nanstd(s)
    1.0
    """
    ...

@disallow("M8", "m8")
@bottleneck_switch(ddof=1)
def nanvar(values, *, axis=..., skipna=..., ddof=..., mask=...):
    """
    Compute the variance along given axis while ignoring NaNs

    Parameters
    ----------
    values : ndarray
    axis : int, optional
    skipna : bool, default True
    ddof : int, default 1
        Delta Degrees of Freedom. The divisor used in calculations is N - ddof,
        where N represents the number of elements.
    mask : ndarray[bool], optional
        nan-mask if known

    Returns
    -------
    result : float
        Unless input is a float array, in which case use the same
        precision as the input array.

    Examples
    --------
    >>> import pandas.core.nanops as nanops
    >>> s = pd.Series([1, np.nan, 2, 3])
    >>> nanops.nanvar(s)
    1.0
    """
    ...

@disallow("M8", "m8")
def nansem(
    values: np.ndarray,
    *,
    axis: int | None = ...,
    skipna: bool = ...,
    ddof: int = ...,
    mask: np.ndarray | None = ...,
) -> float:
    """
    Compute the standard error in the mean along given axis while ignoring NaNs

    Parameters
    ----------
    values : ndarray
    axis : int, optional
    skipna : bool, default True
    ddof : int, default 1
        Delta Degrees of Freedom. The divisor used in calculations is N - ddof,
        where N represents the number of elements.
    mask : ndarray[bool], optional
        nan-mask if known

    Returns
    -------
    result : float64
        Unless input is a float array, in which case use the same
        precision as the input array.

    Examples
    --------
    >>> import pandas.core.nanops as nanops
    >>> s = pd.Series([1, np.nan, 2, 3])
    >>> nanops.nansem(s)
     0.5773502691896258
    """
    ...

nanmin = ...
nanmax = ...

@disallow("O")
def nanargmax(
    values: np.ndarray,
    *,
    axis: int | None = ...,
    skipna: bool = ...,
    mask: np.ndarray | None = ...,
) -> int | np.ndarray:
    """
    Parameters
    ----------
    values : ndarray
    axis : int, optional
    skipna : bool, default True
    mask : ndarray[bool], optional
        nan-mask if known

    Returns
    -------
    result : int or ndarray[int]
        The index/indices  of max value in specified axis or -1 in the NA case

    Examples
    --------
    >>> import pandas.core.nanops as nanops
    >>> arr = np.array([1, 2, 3, np.nan, 4])
    >>> nanops.nanargmax(arr)
    4

    >>> arr = np.array(range(12), dtype=np.float64).reshape(4, 3)
    >>> arr[2:, 2] = np.nan
    >>> arr
    array([[ 0.,  1.,  2.],
           [ 3.,  4.,  5.],
           [ 6.,  7., nan],
           [ 9., 10., nan]])
    >>> nanops.nanargmax(arr, axis=1)
    array([2, 2, 1, 1])
    """
    ...

@disallow("O")
def nanargmin(
    values: np.ndarray,
    *,
    axis: int | None = ...,
    skipna: bool = ...,
    mask: np.ndarray | None = ...,
) -> int | np.ndarray:
    """
    Parameters
    ----------
    values : ndarray
    axis : int, optional
    skipna : bool, default True
    mask : ndarray[bool], optional
        nan-mask if known

    Returns
    -------
    result : int or ndarray[int]
        The index/indices of min value in specified axis or -1 in the NA case

    Examples
    --------
    >>> import pandas.core.nanops as nanops
    >>> arr = np.array([1, 2, 3, np.nan, 4])
    >>> nanops.nanargmin(arr)
    0

    >>> arr = np.array(range(12), dtype=np.float64).reshape(4, 3)
    >>> arr[2:, 0] = np.nan
    >>> arr
    array([[ 0.,  1.,  2.],
           [ 3.,  4.,  5.],
           [nan,  7.,  8.],
           [nan, 10., 11.]])
    >>> nanops.nanargmin(arr, axis=1)
    array([0, 0, 1, 1])
    """
    ...

@disallow("M8", "m8")
def nanskew(
    values: np.ndarray,
    *,
    axis: int | None = ...,
    skipna: bool = ...,
    mask: np.ndarray | None = ...,
) -> float:
    """
    Compute the sample skewness.

    The statistic computed here is the adjusted Fisher-Pearson standardized
    moment coefficient G1. The algorithm computes this coefficient directly
    from the second and third central moment.

    Parameters
    ----------
    values : ndarray
    axis : int, optional
    skipna : bool, default True
    mask : ndarray[bool], optional
        nan-mask if known

    Returns
    -------
    result : float64
        Unless input is a float array, in which case use the same
        precision as the input array.

    Examples
    --------
    >>> import pandas.core.nanops as nanops
    >>> s = pd.Series([1, np.nan, 1, 2])
    >>> nanops.nanskew(s)
    1.7320508075688787
    """
    ...

@disallow("M8", "m8")
def nankurt(
    values: np.ndarray,
    *,
    axis: int | None = ...,
    skipna: bool = ...,
    mask: np.ndarray | None = ...,
) -> float:
    """
    Compute the sample excess kurtosis

    The statistic computed here is the adjusted Fisher-Pearson standardized
    moment coefficient G2, computed directly from the second and fourth
    central moment.

    Parameters
    ----------
    values : ndarray
    axis : int, optional
    skipna : bool, default True
    mask : ndarray[bool], optional
        nan-mask if known

    Returns
    -------
    result : float64
        Unless input is a float array, in which case use the same
        precision as the input array.

    Examples
    --------
    >>> import pandas.core.nanops as nanops
    >>> s = pd.Series([1, np.nan, 1, 3, 2])
    >>> nanops.nankurt(s)
    -1.2892561983471076
    """
    ...

@disallow("M8", "m8")
def nanprod(
    values: np.ndarray,
    *,
    axis: int | None = ...,
    skipna: bool = ...,
    min_count: int = ...,
    mask: np.ndarray | None = ...,
) -> float:
    """
    Parameters
    ----------
    values : ndarray[dtype]
    axis : int, optional
    skipna : bool, default True
    min_count: int, default 0
    mask : ndarray[bool], optional
        nan-mask if known

    Returns
    -------
    Dtype
        The product of all elements on a given axis. ( NaNs are treated as 1)

    Examples
    --------
    >>> import pandas.core.nanops as nanops
    >>> s = pd.Series([1, 2, 3, np.nan])
    >>> nanops.nanprod(s)
    6.0
    """
    ...

def check_below_min_count(
    shape: tuple[int, ...], mask: np.ndarray | None, min_count: int
) -> bool:
    """
    Check for the `min_count` keyword. Returns True if below `min_count` (when
    missing value should be returned from the reduction).

    Parameters
    ----------
    shape : tuple
        The shape of the values (`values.shape`).
    mask : ndarray or None
        Boolean numpy array (typically of same shape as `shape`) or None.
    min_count : int
        Keyword passed through from sum/prod call.

    Returns
    -------
    bool
    """
    ...

@disallow("M8", "m8")
def nancorr(
    a: np.ndarray, b: np.ndarray, *, method=..., min_periods: int | None = ...
):  # -> float:
    """
    a, b: ndarrays
    """
    ...

def get_corr_func(method): ...
@disallow("M8", "m8")
def nancov(
    a: np.ndarray, b: np.ndarray, *, min_periods: int | None = ..., ddof: int | None = ...
): ...
def make_nancomp(op): ...

nangt = ...
nange = ...
nanlt = ...
nanle = ...
naneq = ...
nanne = ...

def nanpercentile(
    values: np.ndarray, q: np.ndarray, *, na_value, mask: np.ndarray, interpolation
):
    """
    Wrapper for np.percentile that skips missing values.

    Parameters
    ----------
    values : np.ndarray[ndim=2]  over which to find quantiles
    q : np.ndarray[float64] of quantile indices to find
    na_value : scalar
        value to return for empty or all-null values
    mask : ndarray[bool]
        locations in values that should be considered missing
    interpolation : str

    Returns
    -------
    quantiles : scalar or array
    """
    ...

def na_accum_func(values: ArrayLike, accum_func, *, skipna: bool) -> ArrayLike:
    """
    Cumulative function with skipna support.

    Parameters
    ----------
    values : np.ndarray or ExtensionArray
    accum_func : {np.cumprod, np.maximum.accumulate, np.cumsum, np.minimum.accumulate}
    skipna : bool

    Returns
    -------
    np.ndarray or ExtensionArray
    """
    ...
