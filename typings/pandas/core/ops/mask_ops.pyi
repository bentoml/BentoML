import numpy as np
from pandas._libs import missing as libmissing

def kleene_or(
    left: bool | np.ndarray,
    right: bool | np.ndarray,
    left_mask: np.ndarray | None,
    right_mask: np.ndarray | None,
): ...
def kleene_xor(
    left: bool | np.ndarray,
    right: bool | np.ndarray,
    left_mask: np.ndarray | None,
    right_mask: np.ndarray | None,
): ...
