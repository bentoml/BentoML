from typing import Any
import numpy as np
from pandas.core.ops.common import unpack_zerodim_and_defer

class OpsMixin:
    @unpack_zerodim_and_defer("__eq__")
    def __eq__(self, other) -> bool: ...
    @unpack_zerodim_and_defer("__ne__")
    def __ne__(self, other) -> bool: ...
    @unpack_zerodim_and_defer("__lt__")
    def __lt__(self, other) -> bool: ...
    @unpack_zerodim_and_defer("__le__")
    def __le__(self, other) -> bool: ...
    @unpack_zerodim_and_defer("__gt__")
    def __gt__(self, other) -> bool: ...
    @unpack_zerodim_and_defer("__ge__")
    def __ge__(self, other) -> bool: ...
    @unpack_zerodim_and_defer("__and__")
    def __and__(self, other): ...
    @unpack_zerodim_and_defer("__rand__")
    def __rand__(self, other): ...
    @unpack_zerodim_and_defer("__or__")
    def __or__(self, other): ...
    @unpack_zerodim_and_defer("__ror__")
    def __ror__(self, other): ...
    @unpack_zerodim_and_defer("__xor__")
    def __xor__(self, other): ...
    @unpack_zerodim_and_defer("__rxor__")
    def __rxor__(self, other): ...
    @unpack_zerodim_and_defer("__add__")
    def __add__(self, other): ...
    @unpack_zerodim_and_defer("__radd__")
    def __radd__(self, other): ...
    @unpack_zerodim_and_defer("__sub__")
    def __sub__(self, other): ...
    @unpack_zerodim_and_defer("__rsub__")
    def __rsub__(self, other): ...
    @unpack_zerodim_and_defer("__mul__")
    def __mul__(self, other): ...
    @unpack_zerodim_and_defer("__rmul__")
    def __rmul__(self, other): ...
    @unpack_zerodim_and_defer("__truediv__")
    def __truediv__(self, other): ...
    @unpack_zerodim_and_defer("__rtruediv__")
    def __rtruediv__(self, other): ...
    @unpack_zerodim_and_defer("__floordiv__")
    def __floordiv__(self, other): ...
    @unpack_zerodim_and_defer("__rfloordiv")
    def __rfloordiv__(self, other): ...
    @unpack_zerodim_and_defer("__mod__")
    def __mod__(self, other): ...
    @unpack_zerodim_and_defer("__rmod__")
    def __rmod__(self, other): ...
    @unpack_zerodim_and_defer("__divmod__")
    def __divmod__(self, other): ...
    @unpack_zerodim_and_defer("__rdivmod__")
    def __rdivmod__(self, other): ...
    @unpack_zerodim_and_defer("__pow__")
    def __pow__(self, other): ...
    @unpack_zerodim_and_defer("__rpow__")
    def __rpow__(self, other): ...

def array_ufunc(self, ufunc: np.ufunc, method: str, *inputs: Any, **kwargs: Any): ...
