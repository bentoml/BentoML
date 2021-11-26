from typing import Any
from pandas._typing import ArrayLike

def should_extension_dispatch(left: ArrayLike, right: Any) -> bool: ...
