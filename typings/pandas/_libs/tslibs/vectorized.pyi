from datetime import tzinfo

import numpy as np
from pandas._libs.tslibs.dtypes import Resolution
from pandas._libs.tslibs.offsets import BaseOffset

"""
For cython types that cannot be represented precisely, closest-available
python equivalents are used, and the precise types kept as adjacent comments.
"""

def dt64arr_to_periodarr(
    stamps: np.ndarray, freq: int, tz: tzinfo | None
) -> np.ndarray: ...
def is_date_array_normalized(stamps: np.ndarray, tz: tzinfo | None = ...) -> bool: ...
def normalize_i8_timestamps(stamps: np.ndarray, tz: tzinfo | None) -> np.ndarray: ...
def get_resolution(stamps: np.ndarray, tz: tzinfo | None = ...) -> Resolution: ...
def ints_to_pydatetime(
    arr: np.ndarray,
    tz: tzinfo | None = ...,
    freq: str | BaseOffset | None = ...,
    fold: bool = ...,
    box: str = ...,
) -> np.ndarray: ...
