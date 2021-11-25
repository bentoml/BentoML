from typing import Dict, Optional

def generate_online_numba_ewma_func(
    engine_kwargs: Optional[Dict[str, bool]]
):  # -> (*args: Unknown, **kwargs: Unknown) -> Unknown | (values: ndarray, deltas: ndarray, minimum_periods: int, old_wt_factor: float, new_wt: float, old_wt: ndarray, adjust: bool, ignore_na: bool) -> tuple[ndarray, ndarray]:
    """
    Generate a numba jitted groupby ewma function specified by values
    from engine_kwargs.
    Parameters
    ----------
    engine_kwargs : dict
        dictionary of arguments to be passed into numba.jit
    Returns
    -------
    Numba function
    """
    ...

class EWMMeanState:
    def __init__(self, com, adjust, ignore_na, axis, shape) -> None: ...
    def run_ewm(self, weighted_avg, deltas, min_periods, ewm_func): ...
    def reset(self): ...
