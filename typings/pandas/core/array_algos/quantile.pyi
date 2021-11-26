from typing import TYPE_CHECKING
import numpy as np
from pandas._typing import ArrayLike

if TYPE_CHECKING: ...

def quantile_compat(
    values: ArrayLike, qs: np.ndarray, interpolation: str
) -> ArrayLike: ...
