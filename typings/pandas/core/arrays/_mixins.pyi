from typing import Any, Sequence

import numpy as np
from pandas._libs.arrays import NDArrayBacked
from pandas._typing import F, PositionalIndexer2D
from pandas.core.array_algos.transforms import shift
from pandas.core.arrays.base import ExtensionArray
from pandas.util._decorators import doc

NDArrayBackedExtensionArrayT = ...

def ravel_compat(meth: F) -> F:
    """
    Decorator to ravel a 2D array before passing it to a cython operation,
    then reshape the result to our own shape.
    """
    ...

class NDArrayBackedExtensionArray(NDArrayBacked, ExtensionArray):
    """
    ExtensionArray that is backed by a single NumPy ndarray.
    """

    _ndarray: np.ndarray
    def take(
        self: NDArrayBackedExtensionArrayT,
        indices: Sequence[int],
        *,
        allow_fill: bool = ...,
        fill_value: Any = ...,
        axis: int = ...
    ) -> NDArrayBackedExtensionArrayT: ...
    def equals(self, other) -> bool: ...
    def argmin(self, axis: int = ..., skipna: bool = ...): ...
    def argmax(self, axis: int = ..., skipna: bool = ...): ...
    def unique(self: NDArrayBackedExtensionArrayT) -> NDArrayBackedExtensionArrayT: ...
    @doc(ExtensionArray.searchsorted)
    def searchsorted(self, value, side=..., sorter=...): ...
    @doc(ExtensionArray.shift)
    def shift(self, periods=..., fill_value=..., axis=...): ...
    def __setitem__(self, key, value): ...
    def __getitem__(
        self: NDArrayBackedExtensionArrayT, key: PositionalIndexer2D
    ) -> NDArrayBackedExtensionArrayT | Any: ...
    @doc(ExtensionArray.fillna)
    def fillna(
        self: NDArrayBackedExtensionArrayT, value=..., method=..., limit=...
    ) -> NDArrayBackedExtensionArrayT: ...
    def __repr__(self) -> str: ...
    def putmask(self: NDArrayBackedExtensionArrayT, mask: np.ndarray, value) -> None:
        """
        Analogue to np.putmask(self, mask, value)

        Parameters
        ----------
        mask : np.ndarray[bool]
        value : scalar or listlike

        Raises
        ------
        TypeError
            If value cannot be cast to self.dtype.
        """
        ...
    def where(
        self: NDArrayBackedExtensionArrayT, mask: np.ndarray, value
    ) -> NDArrayBackedExtensionArrayT:
        """
        Analogue to np.where(mask, self, value)

        Parameters
        ----------
        mask : np.ndarray[bool]
        value : scalar or listlike

        Raises
        ------
        TypeError
            If value cannot be cast to self.dtype.
        """
        ...
    def insert(
        self: NDArrayBackedExtensionArrayT, loc: int, item
    ) -> NDArrayBackedExtensionArrayT:
        """
        Make new ExtensionArray inserting new item at location. Follows
        Python list.append semantics for negative values.

        Parameters
        ----------
        loc : int
        item : object

        Returns
        -------
        type(self)
        """
        ...
    def value_counts(self, dropna: bool = ...):  # -> Series:
        """
        Return a Series containing counts of unique values.

        Parameters
        ----------
        dropna : bool, default True
            Don't include counts of NA values.

        Returns
        -------
        Series
        """
        ...
