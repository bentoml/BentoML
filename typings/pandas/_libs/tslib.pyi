from datetime import tzinfo

import numpy as np

def format_array_from_datetime(
    values: np.ndarray,
    tz: tzinfo | None = ...,
    format: str | None = ...,
    na_rep: object = ...,
) -> np.ndarray: ...
def array_with_unit_to_datetime(
    values: np.ndarray, unit: str, errors: str = ...
) -> tuple[np.ndarray, tzinfo | None]: ...
def array_to_datetime(
    values: np.ndarray,
    errors: str = ...,
    dayfirst: bool = ...,
    yearfirst: bool = ...,
    utc: bool = ...,
    require_iso8601: bool = ...,
    allow_mixed: bool = ...,
) -> tuple[np.ndarray, tzinfo | None]: ...
