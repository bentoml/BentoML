from typing import Literal

import numpy as np

def group_median_float64(
    out: np.ndarray,
    counts: np.ndarray,
    values: np.ndarray,
    labels: np.ndarray,
    min_count: int = ...,
) -> None: ...
def group_cumprod_float64(
    out: np.ndarray,
    values: np.ndarray,
    labels: np.ndarray,
    ngroups: int,
    is_datetimelike: bool,
    skipna: bool = ...,
) -> None: ...
def group_cumsum(
    out: np.ndarray,
    values: np.ndarray,
    labels: np.ndarray,
    ngroups: int,
    is_datetimelike: bool,
    skipna: bool = ...,
) -> None: ...
def group_shift_indexer(
    out: np.ndarray, labels: np.ndarray, ngroups: int, periods: int
) -> None: ...
def group_fillna_indexer(
    out: np.ndarray,
    labels: np.ndarray,
    mask: np.ndarray,
    direction: Literal["ffill", "bfill"],
    limit: int,
    dropna: bool,
) -> None: ...
def group_any_all(
    out: np.ndarray,
    values: np.ndarray,
    labels: np.ndarray,
    mask: np.ndarray,
    val_test: Literal["any", "all"],
    skipna: bool,
) -> None: ...
def group_add(
    out: np.ndarray,
    counts: np.ndarray,
    values: np.ndarray,
    labels: np.ndarray,
    min_count: int = ...,
) -> None: ...
def group_prod(
    out: np.ndarray,
    counts: np.ndarray,
    values: np.ndarray,
    labels: np.ndarray,
    min_count: int = ...,
) -> None: ...
def group_var(
    out: np.ndarray,
    counts: np.ndarray,
    values: np.ndarray,
    labels: np.ndarray,
    min_count: int = ...,
    ddof: int = ...,
) -> None: ...
def group_mean(
    out: np.ndarray,
    counts: np.ndarray,
    values: np.ndarray,
    labels: np.ndarray,
    min_count: int = ...,
    is_datetimelike: bool = ...,
    mask: np.ndarray | None = ...,
    result_mask: np.ndarray | None = ...,
) -> None: ...
def group_ohlc(
    out: np.ndarray,
    counts: np.ndarray,
    values: np.ndarray,
    labels: np.ndarray,
    min_count: int = ...,
) -> None: ...
def group_quantile(
    out: np.ndarray,
    values: np.ndarray,
    labels: np.ndarray,
    mask: np.ndarray,
    q: float,
    interpolation: Literal["linear", "lower", "higher", "nearest", "midpoint"],
) -> None: ...
def group_last(
    out: np.ndarray,
    counts: np.ndarray,
    values: np.ndarray,
    labels: np.ndarray,
    min_count: int = ...,
) -> None: ...
def group_nth(
    out: np.ndarray,
    counts: np.ndarray,
    values: np.ndarray,
    labels: np.ndarray,
    min_count: int = ...,
    rank: int = ...,
) -> None: ...
def group_rank(
    out: np.ndarray,
    values: np.ndarray,
    labels: np.ndarray,
    ngroups: int,
    is_datetimelike: bool,
    ties_method: Literal["aveage", "min", "max", "first", "dense"] = ...,
    ascending: bool = ...,
    pct: bool = ...,
    na_option: Literal["keep", "top", "bottom"] = ...,
) -> None: ...
def group_max(
    out: np.ndarray,
    counts: np.ndarray,
    values: np.ndarray,
    labels: np.ndarray,
    min_count: int = ...,
) -> None: ...
def group_min(
    out: np.ndarray,
    counts: np.ndarray,
    values: np.ndarray,
    labels: np.ndarray,
    min_count: int = ...,
) -> None: ...
def group_cummin(
    out: np.ndarray,
    values: np.ndarray,
    labels: np.ndarray,
    ngroups: int,
    is_datetimelike: bool,
) -> None: ...
def group_cummax(
    out: np.ndarray,
    values: np.ndarray,
    labels: np.ndarray,
    ngroups: int,
    is_datetimelike: bool,
) -> None: ...
