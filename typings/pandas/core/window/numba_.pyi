import functools
from typing import Any, Callable
import numpy as np
from pandas._typing import Scalar

def generate_numba_apply_func(
    args: tuple,
    kwargs: dict[str, Any],
    func: Callable[..., Scalar],
    engine_kwargs: dict[str, bool] | None,
    name: str,
): ...
def generate_numba_ewma_func(
    engine_kwargs: dict[str, bool] | None,
    com: float,
    adjust: bool,
    ignore_na: bool,
    deltas: np.ndarray,
): ...
def generate_numba_table_func(
    args: tuple,
    kwargs: dict[str, Any],
    func: Callable[..., np.ndarray],
    engine_kwargs: dict[str, bool] | None,
    name: str,
): ...
