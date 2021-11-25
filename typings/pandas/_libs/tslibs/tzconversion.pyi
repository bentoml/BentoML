from datetime import timedelta, tzinfo
from typing import Iterable

import numpy as np

def tz_convert_from_utc(vals: np.ndarray, tz: tzinfo) -> np.ndarray: ...
def tz_convert_from_utc_single(val: np.int64, tz: tzinfo) -> np.int64: ...
def tz_localize_to_utc(
    vals: np.ndarray,
    tz: tzinfo | None,
    ambiguous: str | bool | Iterable[bool] | None = ...,
    nonexistent: str | timedelta | np.timedelta64 | None = ...,
) -> np.ndarray: ...
