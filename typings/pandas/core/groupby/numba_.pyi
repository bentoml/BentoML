from typing import Any, Callable

import numpy as np
from pandas._typing import Scalar

"""Common utilities for Numba operations with groupby ops"""

def validate_udf(func: Callable) -> None:
    """
    Validate user defined function for ops when using Numba with groupby ops.

    The first signature arguments should include:

    def f(values, index, ...):
        ...

    Parameters
    ----------
    func : function, default False
        user defined function

    Returns
    -------
    None

    Raises
    ------
    NumbaUtilError
    """
    ...

def generate_numba_agg_func(
    kwargs: dict[str, Any],
    func: Callable[..., Scalar],
    engine_kwargs: dict[str, bool] | None,
) -> Callable[
    [np.ndarray, np.ndarray, np.ndarray, np.ndarray, int, int, Any], np.ndarray
]:
    """
    Generate a numba jitted agg function specified by values from engine_kwargs.

    1. jit the user's function
    2. Return a groupby agg function with the jitted function inline

    Configurations specified in engine_kwargs apply to both the user's
    function _AND_ the groupby evaluation loop.

    Parameters
    ----------
    kwargs : dict
        **kwargs to be passed into the function
    func : function
        function to be applied to each window and will be JITed
    engine_kwargs : dict
        dictionary of arguments to be passed into numba.jit

    Returns
    -------
    Numba function
    """
    ...

def generate_numba_transform_func(
    kwargs: dict[str, Any],
    func: Callable[..., np.ndarray],
    engine_kwargs: dict[str, bool] | None,
) -> Callable[
    [np.ndarray, np.ndarray, np.ndarray, np.ndarray, int, int, Any], np.ndarray
]:
    """
    Generate a numba jitted transform function specified by values from engine_kwargs.

    1. jit the user's function
    2. Return a groupby transform function with the jitted function inline

    Configurations specified in engine_kwargs apply to both the user's
    function _AND_ the groupby evaluation loop.

    Parameters
    ----------
    kwargs : dict
        **kwargs to be passed into the function
    func : function
        function to be applied to each window and will be JITed
    engine_kwargs : dict
        dictionary of arguments to be passed into numba.jit

    Returns
    -------
    Numba function
    """
    ...
