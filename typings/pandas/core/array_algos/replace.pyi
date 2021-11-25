import re
from typing import Any, Pattern

import numpy as np
from pandas._typing import ArrayLike, Scalar

"""
Methods used by Block.replace and related methods.
"""

def should_use_regex(regex: bool, to_replace: Any) -> bool:
    """
    Decide whether to treat `to_replace` as a regular expression.
    """
    ...

def compare_or_regex_search(
    a: ArrayLike, b: Scalar | Pattern, regex: bool, mask: np.ndarray
) -> ArrayLike | bool:
    """
    Compare two array-like inputs of the same shape or two scalar values

    Calls operator.eq or re.search, depending on regex argument. If regex is
    True, perform an element-wise regex matching.

    Parameters
    ----------
    a : array-like
    b : scalar or regex pattern
    regex : bool
    mask : np.ndarray[bool]

    Returns
    -------
    mask : array-like of bool
    """
    ...

def replace_regex(
    values: ArrayLike, rx: re.Pattern, value, mask: np.ndarray | None
):  # -> None:
    """
    Parameters
    ----------
    values : ArrayLike
        Object dtype.
    rx : re.Pattern
    value : Any
    mask : np.ndarray[bool], optional

    Notes
    -----
    Alters values in-place.
    """
    ...
