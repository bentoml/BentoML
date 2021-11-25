from typing import TYPE_CHECKING

import numpy as np
from pandas._typing import ArrayLike

if TYPE_CHECKING: ...

def quantile_compat(values: ArrayLike, qs: np.ndarray, interpolation: str) -> ArrayLike:
    """
    Compute the quantiles of the given values for each quantile in `qs`.

    Parameters
    ----------
    values : np.ndarray or ExtensionArray
    qs : np.ndarray[float64]
    interpolation : str

    Returns
    -------
    np.ndarray or ExtensionArray
    """
    ...
