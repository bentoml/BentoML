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
):  # -> (*args: Unknown, **kwargs: Unknown) -> Unknown | (values: ndarray, begin: ndarray, end: ndarray, minimum_periods: int) -> ndarray:
    """
    Generate a numba jitted apply function specified by values from engine_kwargs.

    1. jit the user's function
    2. Return a rolling apply function with the jitted function inline

    Configurations specified in engine_kwargs apply to both the user's
    function _AND_ the rolling apply function.

    Parameters
    ----------
    args : tuple
        *args to be passed into the function
    kwargs : dict
        **kwargs to be passed into the function
    func : function
        function to be applied to each window and will be JITed
    engine_kwargs : dict
        dictionary of arguments to be passed into numba.jit
    name: str
        name of the caller (Rolling/Expanding)

    Returns
    -------
    Numba function
    """
    ...

def generate_numba_ewma_func(
    engine_kwargs: dict[str, bool] | None,
    com: float,
    adjust: bool,
    ignore_na: bool,
    deltas: np.ndarray,
):  # -> (*args: Unknown, **kwargs: Unknown) -> Unknown | (values: ndarray, begin: ndarray, end: ndarray, minimum_periods: int) -> ndarray:
    """
    Generate a numba jitted ewma function specified by values
    from engine_kwargs.

    Parameters
    ----------
    engine_kwargs : dict
        dictionary of arguments to be passed into numba.jit
    com : float
    adjust : bool
    ignore_na : bool
    deltas : numpy.ndarray

    Returns
    -------
    Numba function
    """
    ...

def generate_numba_table_func(
    args: tuple,
    kwargs: dict[str, Any],
    func: Callable[..., np.ndarray],
    engine_kwargs: dict[str, bool] | None,
    name: str,
):  # -> (*args: Unknown, **kwargs: Unknown) -> Unknown | (values: ndarray, begin: ndarray, end: ndarray, minimum_periods: int) -> Unknown:
    """
    Generate a numba jitted function to apply window calculations table-wise.

    Func will be passed a M window size x N number of columns array, and
    must return a 1 x N number of columns array. Func is intended to operate
    row-wise, but the result will be transposed for axis=1.

    1. jit the user's function
    2. Return a rolling apply function with the jitted function inline

    Parameters
    ----------
    args : tuple
        *args to be passed into the function
    kwargs : dict
        **kwargs to be passed into the function
    func : function
        function to be applied to each window and will be JITed
    engine_kwargs : dict
        dictionary of arguments to be passed into numba.jit
    name : str
        caller (Rolling/Expanding) and original method name for numba cache key

    Returns
    -------
    Numba function
    """
    ...

@functools.lru_cache(maxsize=None)
def generate_manual_numpy_nan_agg_with_axis(nan_func): ...
