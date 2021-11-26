import numpy as np

def unstack(
    values: np.ndarray,
    mask: np.ndarray,
    stride: int,
    length: int,
    width: int,
    new_values: np.ndarray,
    new_mask: np.ndarray,
) -> None: ...
def explode(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]: ...
