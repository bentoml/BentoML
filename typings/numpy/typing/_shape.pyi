import sys
from typing import Sequence, SupportsIndex, Tuple, Union

if sys.version_info >= (3, 8): ...
else: ...
_Shape = Tuple[int, ...]
_ShapeLike = Union[SupportsIndex, Sequence[SupportsIndex]]
