from typing import TYPE_CHECKING

import numpy as np
import pyarrow
from pandas.core.arrays.masked import BaseMaskedArray, BaseMaskedDtype

if TYPE_CHECKING: ...
T = ...

class NumericDtype(BaseMaskedDtype):
    def __from_arrow__(
        self, array: pyarrow.Array | pyarrow.ChunkedArray
    ) -> BaseMaskedArray:
        """
        Construct IntegerArray/FloatingArray from pyarrow Array/ChunkedArray.
        """
        ...

class NumericArray(BaseMaskedArray):
    """
    Base class for IntegerArray and FloatingArray.
    """

    _HANDLED_TYPES = ...
    def __array_ufunc__(self, ufunc: np.ufunc, method: str, *inputs, **kwargs): ...
    def __neg__(self): ...
    def __pos__(self): ...
    def __abs__(self): ...
    def round(self: T, decimals: int = ..., *args, **kwargs) -> T:
        """
        Round each value in the array a to the given number of decimals.

        Parameters
        ----------
        decimals : int, default 0
            Number of decimal places to round to. If decimals is negative,
            it specifies the number of positions to the left of the decimal point.
        *args, **kwargs
            Additional arguments and keywords have no effect but might be
            accepted for compatibility with NumPy.

        Returns
        -------
        NumericArray
            Rounded values of the NumericArray.

        See Also
        --------
        numpy.around : Round values of an np.array.
        DataFrame.round : Round values of a DataFrame.
        Series.round : Round values of a Series.
        """
        ...
