from typing import Any, Callable, Literal

import numpy as np

def roll_sum(
    values: np.ndarray, start: np.ndarray, end: np.ndarray, minp: int
) -> np.ndarray: ...
def roll_mean(
    values: np.ndarray, start: np.ndarray, end: np.ndarray, minp: int
) -> np.ndarray: ...
def roll_var(
    values: np.ndarray, start: np.ndarray, end: np.ndarray, minp: int, ddof: int = ...
) -> np.ndarray: ...
def roll_skew(
    values: np.ndarray, start: np.ndarray, end: np.ndarray, minp: int
) -> np.ndarray: ...
def roll_kurt(
    values: np.ndarray, start: np.ndarray, end: np.ndarray, minp: int
) -> np.ndarray: ...
def roll_median_c(
    values: np.ndarray, start: np.ndarray, end: np.ndarray, minp: int
) -> np.ndarray: ...
def roll_max(
    values: np.ndarray, start: np.ndarray, end: np.ndarray, minp: int
) -> np.ndarray: ...
def roll_min(
    values: np.ndarray, start: np.ndarray, end: np.ndarray, minp: int
) -> np.ndarray: ...
def roll_quantile(
    values: np.ndarray,
    start: np.ndarray,
    end: np.ndarray,
    minp: int,
    quantile: float,
    interpolation: Literal["linear", "lower", "higher", "nearest", "midpoint"],
) -> np.ndarray: ...
def roll_apply(
    obj: object,
    start: np.ndarray,
    end: np.ndarray,
    minp: int,
    function: Callable[..., Any],
    raw: bool,
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
) -> np.ndarray: ...
def roll_weighted_sum(
    values: np.ndarray, weights: np.ndarray, minp: int
) -> np.ndarray: ...
def roll_weighted_mean(
    values: np.ndarray, weights: np.ndarray, minp: int
) -> np.ndarray: ...
def roll_weighted_var(
    values: np.ndarray, weights: np.ndarray, minp: int, ddof: int
) -> np.ndarray: ...
def ewma(
    vals: np.ndarray,
    start: np.ndarray,
    end: np.ndarray,
    minp: int,
    com: float,
    adjust: bool,
    ignore_na: bool,
    deltas: np.ndarray,
) -> np.ndarray: ...
def ewmcov(
    input_x: np.ndarray,
    start: np.ndarray,
    end: np.ndarray,
    minp: int,
    input_y: np.ndarray,
    com: float,
    adjust: bool,
    ignore_na: bool,
    bias: bool,
) -> np.ndarray: ...
