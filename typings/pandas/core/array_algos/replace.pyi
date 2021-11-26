import re
from typing import Any, Pattern
import numpy as np
from pandas._typing import ArrayLike, Scalar

def should_use_regex(regex: bool, to_replace: Any) -> bool: ...
def compare_or_regex_search(
    a: ArrayLike, b: Scalar | Pattern, regex: bool, mask: np.ndarray
) -> ArrayLike | bool: ...
