

import sys
from typing import Optional

"""
Utilities for working with package versions
"""
if sys.version_info < (3, 8):
    ...
else:
    ...
ops = ...
def require_version(requirement: str, hint: Optional[str] = ...) -> None:
    """
    Perform a runtime check of the dependency versions, using the exact same syntax used by pip.

    The installed module version comes from the `site-packages` dir via `importlib_metadata`.

    Args:
        requirement (:obj:`str`): pip style definition, e.g.,  "tokenizers==0.9.4", "tqdm>=4.27", "numpy"
        hint (:obj:`str`, `optional`): what suggestion to print in case of requirements not being met

    Example::

       require_version("pandas>1.1.2")
       require_version("numpy>1.18.5", "this is important to have for whatever reason")

    """
    ...

def require_version_core(requirement): # -> None:
    """require_version wrapper which emits a core-specific hint on failure"""
    ...

