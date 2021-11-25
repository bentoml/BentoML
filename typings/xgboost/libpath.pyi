from typing import List

"""Find the path to xgboost dynamic library files."""

class XGBoostLibraryNotFound(Exception):
    """Error thrown by when xgboost is not found"""

    ...

def find_lib_path() -> List[str]:
    """Find the path to xgboost dynamic library files.

    Returns
    -------
    lib_path
       List of all found library path to xgboost
    """
    ...
