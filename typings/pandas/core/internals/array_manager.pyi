from typing import TYPE_CHECKING, Any, Callable

import numpy as np
from pandas import Float64Index
from pandas._typing import ArrayLike, DtypeObj, Hashable
from pandas.core.arrays import ExtensionArray
from pandas.core.indexes.api import Index
from pandas.core.internals.base import DataManager, SingleDataManager

"""
Experimental manager based on storing a collection of 1D arrays
"""
if TYPE_CHECKING: ...
T = ...

class BaseArrayManager(DataManager):
    """
    Core internal data structure to implement DataFrame and Series.

    Alternative to the BlockManager, storing a list of 1D arrays instead of
    Blocks.

    This is *not* a public API class

    Parameters
    ----------
    arrays : Sequence of arrays
    axes : Sequence of Index
    verify_integrity : bool, default True

    """

    __slots__ = ...
    arrays: list[np.ndarray | ExtensionArray]
    _axes: list[Index]
    def __init__(
        self,
        arrays: list[np.ndarray | ExtensionArray],
        axes: list[Index],
        verify_integrity: bool = ...,
    ) -> None: ...
    def make_empty(self: T, axes=...) -> T:
        """Return an empty ArrayManager with the items axis of len 0 (no columns)"""
        ...
    @property
    def items(self) -> Index: ...
    @property
    def axes(self) -> list[Index]:
        """Axes is BlockManager-compatible order (columns, rows)"""
        ...
    @property
    def shape_proper(self) -> tuple[int, ...]: ...
    def set_axis(self, axis: int, new_labels: Index) -> None: ...
    def consolidate(self: T) -> T: ...
    def is_consolidated(self) -> bool: ...
    def get_dtypes(self): ...
    def __repr__(self) -> str: ...
    def apply(
        self: T,
        f,
        align_keys: list[str] | None = ...,
        ignore_failures: bool = ...,
        **kwargs
    ) -> T:
        """
        Iterate over the arrays, collect and create a new ArrayManager.

        Parameters
        ----------
        f : str or callable
            Name of the Array method to apply.
        align_keys: List[str] or None, default None
        ignore_failures: bool, default False
        **kwargs
            Keywords to pass to `f`

        Returns
        -------
        ArrayManager
        """
        ...
    def apply_with_block(self: T, f, align_keys=..., swap_axis=..., **kwargs) -> T: ...
    def where(self: T, other, cond, align: bool, errors: str) -> T: ...
    def putmask(self, mask, new, align: bool = ...): ...
    def diff(self: T, n: int, axis: int) -> T: ...
    def interpolate(self: T, **kwargs) -> T: ...
    def shift(self: T, periods: int, axis: int, fill_value) -> T: ...
    def fillna(self: T, value, limit, inplace: bool, downcast) -> T: ...
    def downcast(self: T) -> T: ...
    def astype(self: T, dtype, copy: bool = ..., errors: str = ...) -> T: ...
    def convert(
        self: T,
        copy: bool = ...,
        datetime: bool = ...,
        numeric: bool = ...,
        timedelta: bool = ...,
    ) -> T: ...
    def replace(self: T, value, **kwargs) -> T: ...
    def replace_list(
        self: T,
        src_list: list[Any],
        dest_list: list[Any],
        inplace: bool = ...,
        regex: bool = ...,
    ) -> T:
        """do a list replace"""
        ...
    def to_native_types(self, **kwargs): ...
    @property
    def is_mixed_type(self) -> bool: ...
    @property
    def is_numeric_mixed_type(self) -> bool: ...
    @property
    def any_extension_types(self) -> bool:
        """Whether any of the blocks in this manager are extension blocks"""
        ...
    @property
    def is_view(self) -> bool:
        """return a boolean if we are a single block and are a view"""
        ...
    @property
    def is_single_block(self) -> bool: ...
    def get_bool_data(self: T, copy: bool = ...) -> T:
        """
        Select columns that are bool-dtype and object-dtype columns that are all-bool.

        Parameters
        ----------
        copy : bool, default False
            Whether to copy the blocks
        """
        ...
    def get_numeric_data(self: T, copy: bool = ...) -> T:
        """
        Select columns that have a numeric dtype.

        Parameters
        ----------
        copy : bool, default False
            Whether to copy the blocks
        """
        ...
    def copy(self: T, deep=...) -> T:
        """
        Make deep or shallow copy of ArrayManager

        Parameters
        ----------
        deep : bool or string, default True
            If False, return shallow copy (do not copy data)
            If 'all', copy data and a deep copy of the index

        Returns
        -------
        BlockManager
        """
        ...
    def reindex_indexer(
        self: T,
        new_axis,
        indexer,
        axis: int,
        fill_value=...,
        allow_dups: bool = ...,
        copy: bool = ...,
        consolidate: bool = ...,
        only_slice: bool = ...,
        use_na_proxy: bool = ...,
    ) -> T: ...
    def take(self: T, indexer, axis: int = ..., verify: bool = ...) -> T:
        """
        Take items along any axis.
        """
        ...

class ArrayManager(BaseArrayManager):
    ndim = ...
    def __init__(
        self,
        arrays: list[np.ndarray | ExtensionArray],
        axes: list[Index],
        verify_integrity: bool = ...,
    ) -> None: ...
    def fast_xs(self, loc: int) -> ArrayLike:
        """
        Return the array corresponding to `frame.iloc[loc]`.

        Parameters
        ----------
        loc : int

        Returns
        -------
        np.ndarray or ExtensionArray
        """
        ...
    def get_slice(self, slobj: slice, axis: int = ...) -> ArrayManager: ...
    def iget(self, i: int) -> SingleArrayManager:
        """
        Return the data as a SingleArrayManager.
        """
        ...
    def iget_values(self, i: int) -> ArrayLike:
        """
        Return the data for column i as the values (ndarray or ExtensionArray).
        """
        ...
    @property
    def column_arrays(self) -> list[ArrayLike]:
        """
        Used in the JSON C code to access column arrays.
        """
        ...
    def iset(self, loc: int | slice | np.ndarray, value: ArrayLike):  # -> None:
        """
        Set new column(s).

        This changes the ArrayManager in-place, but replaces (an) existing
        column(s), not changing column values in-place).

        Parameters
        ----------
        loc : integer, slice or boolean mask
            Positional location (already bounds checked)
        value : np.ndarray or ExtensionArray
        """
        ...
    def insert(self, loc: int, item: Hashable, value: ArrayLike) -> None:
        """
        Insert item at selected position.

        Parameters
        ----------
        loc : int
        item : hashable
        value : np.ndarray or ExtensionArray
        """
        ...
    def idelete(self, indexer):  # -> Self@ArrayManager:
        """
        Delete selected locations in-place (new block and array, same BlockManager)
        """
        ...
    def grouped_reduce(self: T, func: Callable, ignore_failures: bool = ...) -> T:
        """
        Apply grouped reduction function columnwise, returning a new ArrayManager.

        Parameters
        ----------
        func : grouped reduction function
        ignore_failures : bool, default False
            Whether to drop columns where func raises TypeError.

        Returns
        -------
        ArrayManager
        """
        ...
    def reduce(
        self: T, func: Callable, ignore_failures: bool = ...
    ) -> tuple[T, np.ndarray]:
        """
        Apply reduction function column-wise, returning a single-row ArrayManager.

        Parameters
        ----------
        func : reduction function
        ignore_failures : bool, default False
            Whether to drop columns where func raises TypeError.

        Returns
        -------
        ArrayManager
        np.ndarray
            Indexer of column indices that are retained.
        """
        ...
    def operate_blockwise(self, other: ArrayManager, array_op) -> ArrayManager:
        """
        Apply array_op blockwise with another (aligned) BlockManager.
        """
        ...
    def quantile(
        self,
        *,
        qs: Float64Index,
        axis: int = ...,
        transposed: bool = ...,
        interpolation=...
    ) -> ArrayManager: ...
    def apply_2d(
        self: ArrayManager, f, ignore_failures: bool = ..., **kwargs
    ) -> ArrayManager:
        """
        Variant of `apply`, but where the function should not be applied to
        each column independently, but to the full data as a 2D array.
        """
        ...
    def unstack(self, unstacker, fill_value) -> ArrayManager:
        """
        Return a BlockManager with all blocks unstacked..

        Parameters
        ----------
        unstacker : reshape._Unstacker
        fill_value : Any
            fill_value for newly introduced missing values.

        Returns
        -------
        unstacked : BlockManager
        """
        ...
    def as_array(
        self, transpose: bool = ..., dtype=..., copy: bool = ..., na_value=...
    ) -> np.ndarray:
        """
        Convert the blockmanager data into an numpy array.

        Parameters
        ----------
        transpose : bool, default False
            If True, transpose the return array.
        dtype : object, default None
            Data type of the return array.
        copy : bool, default False
            If True then guarantee that a copy is returned. A value of
            False does not guarantee that the underlying data is not
            copied.
        na_value : object, default lib.no_default
            Value to be used as the missing value sentinel.

        Returns
        -------
        arr : ndarray
        """
        ...

class SingleArrayManager(BaseArrayManager, SingleDataManager):
    __slots__ = ...
    arrays: list[np.ndarray | ExtensionArray]
    _axes: list[Index]
    ndim = ...
    def __init__(
        self,
        arrays: list[np.ndarray | ExtensionArray],
        axes: list[Index],
        verify_integrity: bool = ...,
    ) -> None: ...
    def make_empty(self, axes=...) -> SingleArrayManager:
        """Return an empty ArrayManager with index/array of length 0"""
        ...
    @classmethod
    def from_array(cls, array, index): ...
    @property
    def axes(self): ...
    @property
    def index(self) -> Index: ...
    @property
    def dtype(self): ...
    def external_values(self):  # -> ArrayLike:
        """The array that Series.values returns"""
        ...
    def internal_values(self):
        """The array that Series._values returns"""
        ...
    def array_values(self):  # -> PandasArray:
        """The array that Series.array returns"""
        ...
    @property
    def is_single_block(self) -> bool: ...
    def fast_xs(self, loc: int) -> ArrayLike: ...
    def get_slice(self, slobj: slice, axis: int = ...) -> SingleArrayManager: ...
    def getitem_mgr(self, indexer) -> SingleArrayManager: ...
    def apply(self, func, **kwargs): ...
    def setitem(self, indexer, value): ...
    def idelete(self, indexer) -> SingleArrayManager:
        """
        Delete selected locations in-place (new array, same ArrayManager)
        """
        ...
    def set_values(self, values: ArrayLike):  # -> None:
        """
        Set (replace) the values of the SingleArrayManager in place.

        Use at your own risk! This does not check if the passed values are
        valid for the current SingleArrayManager (length, dtype, etc).
        """
        ...

class NullArrayProxy:
    """
    Proxy object for an all-NA array.

    Only stores the length of the array, and not the dtype. The dtype
    will only be known when actually concatenating (after determining the
    common dtype, for which this proxy is ignored).
    Using this object avoids that the internals/concat.py needs to determine
    the proper dtype and array type.
    """

    ndim = ...
    def __init__(self, n: int) -> None: ...
    @property
    def shape(self): ...
    def to_array(self, dtype: DtypeObj) -> ArrayLike:
        """
        Helper function to create the actual all-NA array from the NullArrayProxy
        object.

        Parameters
        ----------
        arr : NullArrayProxy
        dtype : the dtype for the resulting array

        Returns
        -------
        np.ndarray or ExtensionArray
        """
        ...
