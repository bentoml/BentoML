from typing import Any, Callable
import numpy as np
from pandas._typing import Scalar

def validate_udf(func: Callable) -> None: ...
def generate_numba_agg_func(
    kwargs: dict[str, Any],
    func: Callable[..., Scalar],
    engine_kwargs: dict[str, bool] | None,
) -> Callable[
    [np.ndarray, np.ndarray, np.ndarray, np.ndarray, int, int, Any], np.ndarray
]: ...
def generate_numba_transform_func(
    kwargs: dict[str, Any],
    func: Callable[..., np.ndarray],
    engine_kwargs: dict[str, bool] | None,
) -> Callable[
    [np.ndarray, np.ndarray, np.ndarray, np.ndarray, int, int, Any], np.ndarray
]: ...
