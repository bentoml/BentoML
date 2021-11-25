from typing import Any

import numpy as np
from pandas._typing import ArrayLike

"""
EA-compatible analogue to to np.putmask
"""

def putmask_inplace(values: ArrayLike, mask: np.ndarray, value: Any) -> None:
    """
    ExtensionArray-compatible implementation of np.putmask.  The main
    difference is we do not handle repeating or truncating like numpy.

    Parameters
    ----------
    mask : np.ndarray[bool]
        We assume extract_bool_array has already been called.
    value : Any
    """
    ...

def putmask_smart(values: np.ndarray, mask: np.ndarray, new) -> np.ndarray:
    """
    Return a new ndarray, try to preserve dtype if possible.

    Parameters
    ----------
    values : np.ndarray
        `values`, updated in-place.
    mask : np.ndarray[bool]
        Applies to both sides (array like).
    new : `new values` either scalar or an array like aligned with `values`

    Returns
    -------
    values : ndarray with updated values
        this *may* be a copy of the original

    See Also
    --------
    ndarray.putmask
    """
    ...

def putmask_without_repeat(values: np.ndarray, mask: np.ndarray, new: Any) -> None:
    """
    np.putmask will truncate or repeat if `new` is a listlike with
    len(new) != len(values).  We require an exact match.

    Parameters
    ----------
    values : np.ndarray
    mask : np.ndarray[bool]
    new : Any
    """
    ...

def validate_putmask(values: ArrayLike, mask: np.ndarray) -> tuple[np.ndarray, bool]:
    """
    Validate mask and check if this putmask operation is a no-op.
    """
    ...

def extract_bool_array(mask: ArrayLike) -> np.ndarray:
    """
    If we have a SparseArray or BooleanArray, convert it to ndarray[bool].
    """
    ...

def setitem_datetimelike_compat(
    values: np.ndarray, num_set: int, other
):  # -> list[Unknown]:
    """
    Parameters
    ----------
    values : np.ndarray
    num_set : int
        For putmask, this is mask.sum()
    other : Any
    """
    ...
