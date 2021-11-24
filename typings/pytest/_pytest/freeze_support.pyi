
from typing import List

"""Provides a function to report all internal modules for using freezing
tools."""
def freeze_includes() -> List[str]:
    """Return a list of module names used by pytest that should be
    included by cx_freeze."""
    ...

