
from typing import List, Optional

from _pytest.nodes import Item

"""Utilities for truncating assertion output.

Current default behaviour is to truncate assertion explanations at
~8 terminal lines, unless running in "-vv" mode or running on CI.
"""
DEFAULT_MAX_LINES = ...
DEFAULT_MAX_CHARS = ...
USAGE_MSG = ...
def truncate_if_required(explanation: List[str], item: Item, max_length: Optional[int] = ...) -> List[str]:
    """Truncate this assertion explanation if the given test item is eligible."""
    ...

