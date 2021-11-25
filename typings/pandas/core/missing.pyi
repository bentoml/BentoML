from typing import TYPE_CHECKING, Any

import numpy as np
from pandas import Index
from pandas._typing import ArrayLike, Axis

"""
Routines for filling missing data.
"""
if TYPE_CHECKING: ...

def check_value_size(value, mask: np.ndarray, length: int):
    """
    Validate the size of the values passed to ExtensionArray.fillna.
    """
    ...

def mask_missing(arr: ArrayLike, values_to_mask) -> np.ndarray:
    """
    Return a masking array of same size/shape as arr
    with entries equaling any member of values_to_mask set to True

    Parameters
    ----------
    arr : ArrayLike
    values_to_mask: list, tuple, or scalar

    Returns
    -------
    np.ndarray[bool]
    """
    ...

def clean_fill_method(method, allow_nearest: bool = ...): ...

NP_METHODS = ...
SP_METHODS = ...

def clean_interp_method(method: str, index: Index, **kwargs) -> str: ...
def find_valid_index(values, *, how: str) -> int | None:
    """
    Retrieves the index of the first valid value.

    Parameters
    ----------
    values : ndarray or ExtensionArray
    how : {'first', 'last'}
        Use this parameter to change between the first or last valid index.

    Returns
    -------
    int or None
    """
    ...

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
):  # -> ndarray:
    """
    Wrapper to dispatch to either interpolate_2d or interpolate_2d_with_fill.
    """
    ...

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
) -> np.ndarray:
    """
    Column-wise application of interpolate_1d.

    Notes
    -----
    The signature does differs from interpolate_1d because it only
    includes what is needed for Block.interpolate.
    """
    ...

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
):  # -> ndarray:
    """
    Logic for the 1-d interpolation.  The result should be 1-d, inputs
    xvalues and yvalues will each be 1-d arrays of the same length.

    Bounds_error is currently hardcoded to False since non-scipy ones don't
    take it as an argument.
    """
    ...

def interpolate_2d(
    values,
    method: str = ...,
    axis: Axis = ...,
    limit: int | None = ...,
    limit_area: str | None = ...,
):
    """
    Perform an actual interpolation of values, values will be make 2-d if
    needed fills inplace, returns the result.

    Parameters
    ----------
    values: array-like
        Input array.
    method: str, default "pad"
        Interpolation method. Could be "bfill" or "pad"
    axis: 0 or 1
        Interpolation axis
    limit: int, optional
        Index limit on interpolation.
    limit_area: str, optional
        Limit area for interpolation. Can be "inside" or "outside"

    Returns
    -------
    values: array-like
        Interpolated array.
    """
    ...

_fill_methods = ...

def get_fill_func(method, ndim: int = ...): ...
def clean_reindex_fill_method(method): ...
