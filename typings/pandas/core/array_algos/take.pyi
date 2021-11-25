from typing import TYPE_CHECKING, overload

import numpy as np
from pandas._typing import ArrayLike
from pandas.core.arrays.base import ExtensionArray

if TYPE_CHECKING: ...

@overload
def take_nd(
    arr: np.ndarray, indexer, axis: int = ..., fill_value=..., allow_fill: bool = ...
) -> np.ndarray: ...
@overload
def take_nd(
    arr: ExtensionArray, indexer, axis: int = ..., fill_value=..., allow_fill: bool = ...
) -> ArrayLike: ...
def take_nd(
    arr: ArrayLike, indexer, axis: int = ..., fill_value=..., allow_fill: bool = ...
) -> ArrayLike:
    """
    Specialized Cython take which sets NaN values in one pass

    This dispatches to ``take`` defined on ExtensionArrays. It does not
    currently dispatch to ``SparseArray.take`` for sparse ``arr``.

    Note: this function assumes that the indexer is a valid(ated) indexer with
    no out of bound indices.

    Parameters
    ----------
    arr : np.ndarray or ExtensionArray
        Input array.
    indexer : ndarray
        1-D array of indices to take, subarrays corresponding to -1 value
        indices are filed with fill_value
    axis : int, default 0
        Axis to take from
    fill_value : any, default np.nan
        Fill value to replace -1 values with
    allow_fill : bool, default True
        If False, indexer is assumed to contain no -1 values so no filling
        will be done.  This short-circuits computation of a mask.  Result is
        undefined if allow_fill == False and -1 is present in indexer.

    Returns
    -------
    subarray : np.ndarray or ExtensionArray
        May be the same type as the input, or cast to an ndarray.
    """
    ...

def take_1d(
    arr: ArrayLike, indexer: np.ndarray, fill_value=..., allow_fill: bool = ...
) -> ArrayLike:
    """
    Specialized version for 1D arrays. Differences compared to `take_nd`:

    - Assumes input array has already been converted to numpy array / EA
    - Assumes indexer is already guaranteed to be int64 dtype ndarray
    - Only works for 1D arrays

    To ensure the lowest possible overhead.

    Note: similarly to `take_nd`, this function assumes that the indexer is
    a valid(ated) indexer with no out of bound indices.
    """
    ...

def take_2d_multi(
    arr: np.ndarray, indexer: tuple[np.ndarray, np.ndarray], fill_value=...
) -> np.ndarray:
    """
    Specialized Cython take which sets NaN values in one pass.
    """
    ...

_take_1d_dict = ...
_take_2d_axis0_dict = ...
_take_2d_axis1_dict = ...
_take_2d_multi_dict = ...
