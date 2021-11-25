from typing import Any, Callable, Hashable, Sequence

import numpy as np
from pandas._libs import internals as libinternals
from pandas._typing import ArrayLike, Dtype, DtypeObj, Shape, type_t
from pandas.core.indexes.api import Float64Index, Index
from pandas.core.internals.base import DataManager, SingleDataManager
from pandas.core.internals.blocks import Block

T = ...

class BaseBlockManager(DataManager):
    """
    Core internal data structure to implement DataFrame, Series, etc.

    Manage a bunch of labeled 2D mixed-type ndarrays. Essentially it's a
    lightweight blocked set of labeled data to be manipulated by the DataFrame
    public API class

    Attributes
    ----------
    shape
    ndim
    axes
    values
    items

    Methods
    -------
    set_axis(axis, new_labels)
    copy(deep=True)

    get_dtypes

    apply(func, axes, block_filter_fn)

    get_bool_data
    get_numeric_data

    get_slice(slice_like, axis)
    get(label)
    iget(loc)

    take(indexer, axis)
    reindex_axis(new_labels, axis)
    reindex_indexer(new_labels, indexer, axis)

    delete(label)
    insert(loc, label, value)
    set(label, value)

    Parameters
    ----------
    blocks: Sequence of Block
    axes: Sequence of Index
    verify_integrity: bool, default True

    Notes
    -----
    This is *not* a public API class
    """

    __slots__ = ...
    _blknos: np.ndarray
    _blklocs: np.ndarray
    blocks: tuple[Block, ...]
    axes: list[Index]
    ndim: int
    _known_consolidated: bool
    _is_consolidated: bool
    def __init__(self, blocks, axes, verify_integrity=...) -> None: ...
    @classmethod
    def from_blocks(cls: type_t[T], blocks: list[Block], axes: list[Index]) -> T: ...
    @property
    def blknos(self):  # -> ndarray:
        """
        Suppose we want to find the array corresponding to our i'th column.

        blknos[i] identifies the block from self.blocks that contains this column.

        blklocs[i] identifies the column of interest within
        self.blocks[self.blknos[i]]
        """
        ...
    @property
    def blklocs(self):  # -> ndarray:
        """
        See blknos.__doc__
        """
        ...
    def make_empty(self: T, axes=...) -> T:
        """return an empty BlockManager with the items axis of len 0"""
        ...
    def __nonzero__(self) -> bool: ...
    __bool__ = ...
    def set_axis(self, axis: int, new_labels: Index) -> None: ...
    @property
    def is_single_block(self) -> bool: ...
    @property
    def items(self) -> Index: ...
    def get_dtypes(self): ...
    @property
    def arrays(self) -> list[ArrayLike]:
        """
        Quick access to the backing arrays of the Blocks.

        Only for compatibility with ArrayManager for testing convenience.
        Not to be used in actual code, and return value is not the same as the
        ArrayManager method (list of 1D arrays vs iterator of 2D ndarrays / 1D EAs).
        """
        ...
    def __repr__(self) -> str: ...
    def apply(
        self: T,
        f,
        align_keys: list[str] | None = ...,
        ignore_failures: bool = ...,
        **kwargs
    ) -> T:
        """
        Iterate over the blocks, collect and create a new BlockManager.

        Parameters
        ----------
        f : str or callable
            Name of the Block method to apply.
        align_keys: List[str] or None, default None
        ignore_failures: bool, default False
        **kwargs
            Keywords to pass to `f`

        Returns
        -------
        BlockManager
        """
        ...
    def where(self: T, other, cond, align: bool, errors: str) -> T: ...
    def setitem(self: T, indexer, value) -> T: ...
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
    def replace(self: T, to_replace, value, inplace: bool, regex: bool) -> T: ...
    def replace_list(
        self: T,
        src_list: list[Any],
        dest_list: list[Any],
        inplace: bool = ...,
        regex: bool = ...,
    ) -> T:
        """do a list replace"""
        ...
    def to_native_types(self: T, **kwargs) -> T:
        """
        Convert values to native types (strings / python objects) that are used
        in formatting (repr / csv).
        """
        ...
    def is_consolidated(self) -> bool:
        """
        Return True if more than one block with the same dtype
        """
        ...
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
    def get_bool_data(self: T, copy: bool = ...) -> T:
        """
        Select blocks that are bool-dtype and columns from object-dtype blocks
        that are all-bool.

        Parameters
        ----------
        copy : bool, default False
            Whether to copy the blocks
        """
        ...
    def get_numeric_data(self: T, copy: bool = ...) -> T:
        """
        Parameters
        ----------
        copy : bool, default False
            Whether to copy the blocks
        """
        ...
    @property
    def nblocks(self) -> int: ...
    def copy(self: T, deep=...) -> T:
        """
        Make deep or shallow copy of BlockManager

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
    def consolidate(self: T) -> T:
        """
        Join together blocks having same dtype

        Returns
        -------
        y : BlockManager
        """
        ...
    def reindex_indexer(
        self: T,
        new_axis: Index,
        indexer,
        axis: int,
        fill_value=...,
        allow_dups: bool = ...,
        copy: bool = ...,
        consolidate: bool = ...,
        only_slice: bool = ...,
    ) -> T:
        """
        Parameters
        ----------
        new_axis : Index
        indexer : ndarray of int64 or None
        axis : int
        fill_value : object, default None
        allow_dups : bool, default False
        copy : bool, default True
        consolidate: bool, default True
            Whether to consolidate inplace before reindexing.
        only_slice : bool, default False
            Whether to take views, not copies, along columns.

        pandas-indexer with -1's only.
        """
        ...
    def take(self: T, indexer, axis: int = ..., verify: bool = ...) -> T:
        """
        Take items along any axis.

        indexer : np.ndarray or slice
        axis : int, default 1
        verify : bool, default True
            Check that all entries are between 0 and len(self) - 1, inclusive.
            Pass verify=False if this check has been done by the caller.

        Returns
        -------
        BlockManager
        """
        ...

class BlockManager(libinternals.BlockManager, BaseBlockManager):
    """
    BaseBlockManager that holds 2D blocks.
    """

    ndim = ...
    def __init__(
        self, blocks: Sequence[Block], axes: Sequence[Index], verify_integrity: bool = ...
    ) -> None: ...
    @classmethod
    def from_blocks(cls, blocks: list[Block], axes: list[Index]) -> BlockManager:
        """
        Constructor for BlockManager and SingleBlockManager with same signature.
        """
        ...
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
    def iget(self, i: int) -> SingleBlockManager:
        """
        Return the data as a SingleBlockManager.
        """
        ...
    def iget_values(self, i: int) -> ArrayLike:
        """
        Return the data for column i as the values (ndarray or ExtensionArray).
        """
        ...
    @property
    def column_arrays(self) -> list[np.ndarray]:
        """
        Used in the JSON C code to access column arrays.
        This optimizes compared to using `iget_values` by converting each
        block.values to a np.ndarray only once up front
        """
        ...
    def iset(self, loc: int | slice | np.ndarray, value: ArrayLike):  # -> None:
        """
        Set new item in-place. Does not consolidate. Adds new Block if not
        contained in the current set of items
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
    def idelete(self, indexer) -> BlockManager:
        """
        Delete selected locations, returning a new BlockManager.
        """
        ...
    def grouped_reduce(self: T, func: Callable, ignore_failures: bool = ...) -> T:
        """
        Apply grouped reduction function blockwise, returning a new BlockManager.

        Parameters
        ----------
        func : grouped reduction function
        ignore_failures : bool, default False
            Whether to drop blocks where func raises TypeError.

        Returns
        -------
        BlockManager
        """
        ...
    def reduce(
        self: T, func: Callable, ignore_failures: bool = ...
    ) -> tuple[T, np.ndarray]:
        """
        Apply reduction function blockwise, returning a single-row BlockManager.

        Parameters
        ----------
        func : reduction function
        ignore_failures : bool, default False
            Whether to drop blocks where func raises TypeError.

        Returns
        -------
        BlockManager
        np.ndarray
            Indexer of mgr_locs that are retained.
        """
        ...
    def operate_blockwise(self, other: BlockManager, array_op) -> BlockManager:
        """
        Apply array_op blockwise with another (aligned) BlockManager.
        """
        ...
    def quantile(self: T, *, qs: Float64Index, axis: int = ..., interpolation=...) -> T:
        """
        Iterate over blocks applying quantile reduction.
        This routine is intended for reduction type operations and
        will do inference on the generated blocks.

        Parameters
        ----------
        axis: reduction axis, default 0
        consolidate: bool, default True. Join together blocks having same
            dtype
        interpolation : type of interpolation, default 'linear'
        qs : list of the quantiles to be computed

        Returns
        -------
        BlockManager
        """
        ...
    def unstack(self, unstacker, fill_value) -> BlockManager:
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
    def to_dict(self, copy: bool = ...):  # -> dict[str, Self@BlockManager]:
        """
        Return a dict of str(dtype) -> BlockManager

        Parameters
        ----------
        copy : bool, default True

        Returns
        -------
        values : a dict of dtype -> BlockManager
        """
        ...
    def as_array(
        self,
        transpose: bool = ...,
        dtype: Dtype | None = ...,
        copy: bool = ...,
        na_value=...,
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

class SingleBlockManager(BaseBlockManager, SingleDataManager):
    """manage a single block with"""

    ndim = ...
    _is_consolidated = ...
    _known_consolidated = ...
    __slots__ = ...
    is_single_block = ...
    def __init__(
        self, block: Block, axis: Index, verify_integrity: bool = ..., fastpath=...
    ) -> None: ...
    @classmethod
    def from_blocks(cls, blocks: list[Block], axes: list[Index]) -> SingleBlockManager:
        """
        Constructor for BlockManager and SingleBlockManager with same signature.
        """
        ...
    @classmethod
    def from_array(cls, array: ArrayLike, index: Index) -> SingleBlockManager:
        """
        Constructor for if we have an array that is not yet a Block.
        """
        ...
    def __getstate__(self): ...
    def __setstate__(self, state): ...
    def getitem_mgr(self, indexer) -> SingleBlockManager: ...
    def get_slice(self, slobj: slice, axis: int = ...) -> SingleBlockManager: ...
    @property
    def index(self) -> Index: ...
    @property
    def dtype(self) -> DtypeObj: ...
    def get_dtypes(self) -> np.ndarray: ...
    def external_values(self):  # -> ArrayLike:
        """The array that Series.values returns"""
        ...
    def internal_values(self):  # -> ndarray | ExtensionArray:
        """The array that Series._values returns"""
        ...
    def array_values(self):  # -> ExtensionArray:
        """The array that Series.array returns"""
        ...
    def is_consolidated(self) -> bool: ...
    def idelete(self, indexer) -> SingleBlockManager:
        """
        Delete single location from SingleBlockManager.

        Ensures that self.blocks doesn't become empty.
        """
        ...
    def fast_xs(self, loc):
        """
        fast path for getting a cross-section
        return a view of the data
        """
        ...
    def set_values(self, values: ArrayLike):  # -> None:
        """
        Set the values of the single block in place.

        Use at your own risk! This does not check if the passed values are
        valid for the current Block/SingleBlockManager (length, dtype, etc).
        """
        ...

def create_block_manager_from_blocks(
    blocks: list[Block], axes: list[Index], consolidate: bool = ...
) -> BlockManager: ...
def create_block_manager_from_arrays(
    arrays, names: Index, axes: list[Index], consolidate: bool = ...
) -> BlockManager: ...
def construction_error(
    tot_items: int, block_shape: Shape, axes: list[Index], e: ValueError | None = ...
):  # -> ValueError:
    """raise a helpful message about our construction"""
    ...
