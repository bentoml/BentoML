from typing import TYPE_CHECKING, Any
import numpy as np
from pandas import Index
from pandas._typing import ArrayLike, Axis

if TYPE_CHECKING: ...

def check_value_size(value, mask: np.ndarray, length: int): ...
def mask_missing(arr: ArrayLike, values_to_mask) -> np.ndarray: ...
def clean_fill_method(method, allow_nearest: bool = ...): ...

NP_METHODS = ...
SP_METHODS = ...

def clean_interp_method(method: str, index: Index, **kwargs) -> str: ...
def find_valid_index(values, *, how: str) -> int | None: ...
def interpolate_array_2d(
    data: np.ndarray,
    method: str = ...,
    axis: int = ...,
    index: Index | None = ...,
    limit: int | None = ...,
    limit_direction: str = ...,
    limit_area: str | None = ...,
    fill_value: Any | None = ...,
    coerce: bool = ...,
    downcast: str | None = ...,
    **kwargs
): ...
def interpolate_2d_with_fill(
    data: np.ndarray,
    index: Index,
    axis: int,
    method: str = ...,
    limit: int | None = ...,
    limit_direction: str = ...,
    limit_area: str | None = ...,
    fill_value: Any | None = ...,
    **kwargs
) -> np.ndarray: ...
def interpolate_1d(
    xvalues: Index,
    yvalues: np.ndarray,
    method: str | None = ...,
    limit: int | None = ...,
    limit_direction: str = ...,
    limit_area: str | None = ...,
    fill_value: Any | None = ...,
    bounds_error: bool = ...,
    order: int | None = ...,
    **kwargs
): ...
def interpolate_2d(
    values,
    method: str = ...,
    axis: Axis = ...,
    limit: int | None = ...,
    limit_area: str | None = ...,
): ...

_fill_methods = ...

def get_fill_func(method, ndim: int = ...): ...
def clean_reindex_fill_method(method): ...
