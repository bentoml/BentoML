

from datetime import datetime, tzinfo

import numpy as np

DT64NS_DTYPE: np.dtype
TD64NS_DTYPE: np.dtype
class OutOfBoundsTimedelta(ValueError):
    ...


def precision_from_unit(unit: str) -> tuple[int, int],:
    ...

def ensure_datetime64ns(arr: np.ndarray, copy: bool = ...) -> np.ndarray:
    ...

def ensure_timedelta64ns(arr: np.ndarray, copy: bool = ...) -> np.ndarray:
    ...

def datetime_to_datetime64(values: np.ndarray) -> tuple[np.ndarray, tzinfo | None],:
    ...

def localize_pydatetime(dt: datetime, tz: tzinfo | None) -> datetime:
    ...

