from typing import Callable

"""Common utilities for Numba operations"""
GLOBAL_USE_NUMBA: bool = ...
NUMBA_FUNC_CACHE: dict[tuple[Callable, str], Callable] = ...

def maybe_use_numba(engine: str | None) -> bool:
    """Signal whether to use numba routines."""
    ...

def set_use_numba(enable: bool = ...) -> None: ...
def get_jit_arguments(
    engine_kwargs: dict[str, bool] | None = ..., kwargs: dict | None = ...
) -> tuple[bool, bool, bool]:
    """
    Return arguments to pass to numba.JIT, falling back on pandas default JIT settings.

    Parameters
    ----------
    engine_kwargs : dict, default None
        user passed keyword arguments for numba.JIT
    kwargs : dict, default None
        user passed keyword arguments to pass into the JITed function

    Returns
    -------
    (bool, bool, bool)
        nopython, nogil, parallel

    Raises
    ------
    NumbaUtilError
    """
    ...

def jit_user_function(
    func: Callable, nopython: bool, nogil: bool, parallel: bool
) -> Callable:
    """
    JIT the user's function given the configurable arguments.

    Parameters
    ----------
    func : function
        user defined function
    nopython : bool
        nopython parameter for numba.JIT
    nogil : bool
        nogil parameter for numba.JIT
    parallel : bool
        parallel parameter for numba.JIT

    Returns
    -------
    function
        Numba JITed function
    """
    ...
