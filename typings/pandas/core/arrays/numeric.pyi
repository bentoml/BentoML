from typing import TYPE_CHECKING
import numpy as np
import pyarrow
from pandas.core.arrays.masked import BaseMaskedArray, BaseMaskedDtype

if TYPE_CHECKING: ...
T = ...

class NumericDtype(BaseMaskedDtype):
    def __from_arrow__(
        self, array: pyarrow.Array | pyarrow.ChunkedArray
    ) -> BaseMaskedArray: ...

class NumericArray(BaseMaskedArray):
    _HANDLED_TYPES = ...
    def __array_ufunc__(self, ufunc: np.ufunc, method: str, *inputs, **kwargs): ...
    def __neg__(self): ...
    def __pos__(self): ...
    def __abs__(self): ...
    def round(self: T, decimals: int = ..., *args, **kwargs) -> T: ...
