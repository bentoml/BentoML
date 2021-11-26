from typing import Callable
from pandas._typing import F

def unpack_zerodim_and_defer(name: str) -> Callable[[F], F]: ...
