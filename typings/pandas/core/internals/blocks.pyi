from typing import TYPE_CHECKING, Any, Callable

import numpy as np
from pandas import Float64Index, Index
from pandas._libs import internals as libinternals
from pandas._libs.internals import BlockPlacement
from pandas._typing import ArrayLike, Dtype, DtypeObj, F, Shape, final
from pandas.core.arrays import DatetimeArray, ExtensionArray, TimedeltaArray
from pandas.core.arrays._mixins import NDArrayBackedExtensionArray
from pandas.core.base import PandasObject
from pandas.util._decorators import cache_readonly

if TYPE_CHECKING: ...
_dtype_obj = ...

def maybe_split(meth: F) -> F:
    """
    If we have a multi-column block, split and operate block-wise.  Otherwise
    use the original method.
    """
    ...

class Block(PandasObject):
    """
    Canonical n-dimensional unit of homogeneous dtype contained in a pandas
    data structure

    Index-ignorant; let the container take care of that
    """

    values: np.ndarray | ExtensionArray
    ndim: int
    __init__: Callable
    __slots__ = ...
    is_numeric = ...
    is_object = ...
    is_extension = ...
    _can_consolidate = ...
    _validate_ndim = ...
    @property
    def is_view(self) -> bool:
        """return a boolean if I am possibly a view"""
        ...
    @final
    @cache_readonly
    def is_categorical(self) -> bool: ...
    @final
    @property
    def is_bool(self) -> bool:
        """
        We can be bool if a) we are bool dtype or b) object dtype with bool objects.
        """
        ...
    @final
    def external_values(self): ...
    @property
    def array_values(self) -> ExtensionArray:
        """
        The array that Series.array returns. Always an ExtensionArray.
        """
        ...
    def get_values(self, dtype: DtypeObj | None = ...) -> np.ndarray:
        """
        return an internal format, currently just the ndarray
        this is often overridden to handle to_dense like operations
        """
        ...
    @final
    @cache_readonly
    def fill_value(self): ...
    @property
    def mgr_locs(self) -> BlockPlacement: ...
    @mgr_locs.setter
    def mgr_locs(self, new_mgr_locs: BlockPlacement): ...
    @final
    def make_block(self, values, placement=...) -> Block:
        """
        Create a new block, with type inference propagate any values that are
        not specified
        """
        ...
    @final
    def make_block_same_class(
        self, values, placement: BlockPlacement | None = ...
    ) -> Block:
        """Wrap given values in a block of same type as self."""
        ...
    @final
    def __repr__(self) -> str: ...
    @final
    def __len__(self) -> int: ...
    @final
    def getitem_block(self, slicer) -> Block:
        """
        Perform __getitem__-like, return result as block.

        Only supports slices that preserve dimensionality.
        """
        ...
    def getitem_block_index(self, slicer: slice) -> Block:
        """
        Perform __getitem__-like specialized to slicing along index.

        Assumes self.ndim == 2
        """
        ...
    @final
    def getitem_block_columns(self, slicer, new_mgr_locs: BlockPlacement) -> Block:
        """
        Perform __getitem__-like, return result as block.

        Only supports slices that preserve dimensionality.
        """
        ...
    @property
    def shape(self) -> Shape: ...
    @cache_readonly
    def dtype(self) -> DtypeObj: ...
    def iget(self, i): ...
    def set_inplace(self, locs, values):  # -> None:
        """
        Modify block values in-place with new item value.

        Notes
        -----
        `set` never creates a new array or new Block, whereas `setitem` _may_
        create a new array and always creates a new Block.
        """
        ...
    def delete(self, loc) -> None:
        """
        Delete given loc(-s) from block in-place.
        """
        ...
    @final
    def apply(self, func, **kwargs) -> list[Block]:
        """
        apply the function to my values; return a block if we are not
        one
        """
        ...
    def reduce(self, func, ignore_failures: bool = ...) -> list[Block]: ...
    def fillna(self, value, limit=..., inplace: bool = ..., downcast=...) -> list[Block]:
        """
        fillna on the block with the value. If we fail, then convert to
        ObjectBlock and try again
        """
        ...
    @final
    def split_and_operate(self, func, *args, **kwargs) -> list[Block]:
        """
        Split the block and apply func column-by-column.

        Parameters
        ----------
        func : Block method
        *args
        **kwargs

        Returns
        -------
        List[Block]
        """
        ...
    @final
    def downcast(self, dtypes=...) -> list[Block]:
        """try to downcast each item to the dict of dtypes if present"""
        ...
    @final
    def astype(self, dtype, copy: bool = ..., errors: str = ...):  # -> Block:
        """
        Coerce to the new dtype.

        Parameters
        ----------
        dtype : str, dtype convertible
        copy : bool, default False
            copy if indicated
        errors : str, {'raise', 'ignore'}, default 'raise'
            - ``raise`` : allow exceptions to be raised
            - ``ignore`` : suppress exceptions. On error return original object

        Returns
        -------
        Block
        """
        ...
    def convert(
        self,
        copy: bool = ...,
        datetime: bool = ...,
        numeric: bool = ...,
        timedelta: bool = ...,
    ) -> list[Block]:
        """
        attempt to coerce any object types to better types return a copy
        of the block (if copy = True) by definition we are not an ObjectBlock
        here!
        """
        ...
    @final
    def should_store(self, value: ArrayLike) -> bool:
        """
        Should we set self.values[indexer] = value inplace or do we need to cast?

        Parameters
        ----------
        value : np.ndarray or ExtensionArray

        Returns
        -------
        bool
        """
        ...
    @final
    def to_native_types(self, na_rep=..., quoting=..., **kwargs):  # -> Block:
        """convert to our native types format"""
        ...
    @final
    def copy(self, deep: bool = ...):  # -> Block:
        """copy constructor"""
        ...
    @final
    def replace(
        self, to_replace, value, inplace: bool = ..., regex: bool = ...
    ) -> list[Block]:
        """
        replace the to_replace value with value, possible to create new
        blocks here this is just a call to putmask. regex is not used here.
        It is used in ObjectBlocks.  It is here for API compatibility.
        """
        ...
    def setitem(self, indexer, value):
        """
        Attempt self.values[indexer] = value, possibly creating a new array.

        Parameters
        ----------
        indexer : tuple, list-like, array-like, slice
            The subset of self.values to set
        value : object
            The value being set

        Returns
        -------
        Block

        Notes
        -----
        `indexer` is a direct slice/positional indexer. `value` must
        be a compatible shape.
        """
        ...
    def putmask(self, mask, new) -> list[Block]:
        """
        putmask the data to the block; it is possible that we may create a
        new dtype of block

        Return the resulting block(s).

        Parameters
        ----------
        mask : np.ndarray[bool], SparseArray[bool], or BooleanArray
        new : a ndarray/object

        Returns
        -------
        List[Block]
        """
        ...
    @final
    def coerce_to_target_dtype(self, other) -> Block:
        """
        coerce the current block to a dtype compat for other
        we will return a block, possibly object, and not raise

        we can also safely try to coerce to the same dtype
        and will receive the same block
        """
        ...
    def interpolate(
        self,
        method: str = ...,
        axis: int = ...,
        index: Index | None = ...,
        inplace: bool = ...,
        limit: int | None = ...,
        limit_direction: str = ...,
        limit_area: str | None = ...,
        fill_value: Any | None = ...,
        coerce: bool = ...,
        downcast: str | None = ...,
        **kwargs
    ) -> list[Block]: ...
    def take_nd(
        self,
        indexer,
        axis: int,
        new_mgr_locs: BlockPlacement | None = ...,
        fill_value=...,
    ) -> Block:
        """
        Take values according to indexer and return them as a block.bb

        """
        ...
    def diff(self, n: int, axis: int = ...) -> list[Block]:
        """return block for the diff of the values"""
        ...
    def shift(self, periods: int, axis: int = ..., fill_value: Any = ...) -> list[Block]:
        """shift the block by periods, possibly upcast"""
        ...
    def where(self, other, cond, errors=...) -> list[Block]:
        """
        evaluate the block; return result block(s) from the result

        Parameters
        ----------
        other : a ndarray/object
        cond : np.ndarray[bool], SparseArray[bool], or BooleanArray
        errors : str, {'raise', 'ignore'}, default 'raise'
            - ``raise`` : allow exceptions to be raised
            - ``ignore`` : suppress exceptions. On error return original object

        Returns
        -------
        List[Block]
        """
        ...
    @final
    def quantile(self, qs: Float64Index, interpolation=..., axis: int = ...) -> Block:
        """
        compute the quantiles of the

        Parameters
        ----------
        qs : Float64Index
            List of the quantiles to be computed.
        interpolation : str, default 'linear'
            Type of interpolation.
        axis : int, default 0
            Axis to compute.

        Returns
        -------
        Block
        """
        ...

class EABackedBlock(Block):
    """
    Mixin for Block subclasses backed by ExtensionArray.
    """

    values: ExtensionArray
    def delete(self, loc) -> None:
        """
        Delete given loc(-s) from block in-place.
        """
        ...
    @cache_readonly
    def array_values(self) -> ExtensionArray: ...
    def get_values(self, dtype: DtypeObj | None = ...) -> np.ndarray:
        """
        return object dtype as boxed values, such as Timestamps/Timedelta
        """
        ...
    def interpolate(
        self, method=..., axis=..., inplace=..., limit=..., fill_value=..., **kwargs
    ): ...

class ExtensionBlock(libinternals.Block, EABackedBlock):
    """
    Block for holding extension types.

    Notes
    -----
    This holds all 3rd-party extension array types. It's also the immediate
    parent class for our internal extension types' blocks, CategoricalBlock.

    ExtensionArrays are limited to 1-D.
    """

    _can_consolidate = ...
    _validate_ndim = ...
    is_extension = ...
    values: ExtensionArray
    @cache_readonly
    def shape(self) -> Shape: ...
    def iget(self, col): ...
    def set_inplace(self, locs, values): ...
    def putmask(self, mask, new) -> list[Block]:
        """
        See Block.putmask.__doc__
        """
        ...
    @property
    def is_view(self) -> bool:
        """Extension arrays are never treated as views."""
        ...
    @cache_readonly
    def is_numeric(self): ...
    def setitem(self, indexer, value):  # -> Self@ExtensionBlock:
        """
        Attempt self.values[indexer] = value, possibly creating a new array.

        This differs from Block.setitem by not allowing setitem to change
        the dtype of the Block.

        Parameters
        ----------
        indexer : tuple, list-like, array-like, slice
            The subset of self.values to set
        value : object
            The value being set

        Returns
        -------
        Block

        Notes
        -----
        `indexer` is a direct slice/positional indexer. `value` must
        be a compatible shape.
        """
        ...
    def take_nd(
        self,
        indexer,
        axis: int = ...,
        new_mgr_locs: BlockPlacement | None = ...,
        fill_value=...,
    ) -> Block:
        """
        Take values according to indexer and return them as a block.
        """
        ...
    @final
    def getitem_block_index(self, slicer: slice) -> ExtensionBlock:
        """
        Perform __getitem__-like specialized to slicing along index.
        """
        ...
    def fillna(
        self, value, limit=..., inplace: bool = ..., downcast=...
    ) -> list[Block]: ...
    def diff(self, n: int, axis: int = ...) -> list[Block]: ...
    def shift(self, periods: int, axis: int = ..., fill_value: Any = ...) -> list[Block]:
        """
        Shift the block by `periods`.

        Dispatches to underlying ExtensionArray and re-boxes in an
        ExtensionBlock.
        """
        ...
    def where(self, other, cond, errors=...) -> list[Block]: ...

class NumpyBlock(libinternals.NumpyBlock, Block):
    values: np.ndarray
    getitem_block_index = ...

class NumericBlock(NumpyBlock):
    __slots__ = ...
    is_numeric = ...

class NDArrayBackedExtensionBlock(libinternals.NDArrayBackedBlock, EABackedBlock):
    """
    Block backed by an NDArrayBackedExtensionArray
    """

    values: NDArrayBackedExtensionArray
    getitem_block_index = ...
    @property
    def is_view(self) -> bool:
        """return a boolean if I am possibly a view"""
        ...
    def setitem(self, indexer, value): ...
    def putmask(self, mask, new) -> list[Block]: ...
    def where(self, other, cond, errors=...) -> list[Block]: ...
    def diff(self, n: int, axis: int = ...) -> list[Block]:
        """
        1st discrete difference.

        Parameters
        ----------
        n : int
            Number of periods to diff.
        axis : int, default 0
            Axis to diff upon.

        Returns
        -------
        A list with a new Block.

        Notes
        -----
        The arguments here are mimicking shift so they are called correctly
        by apply.
        """
        ...
    def shift(
        self, periods: int, axis: int = ..., fill_value: Any = ...
    ) -> list[Block]: ...
    def fillna(
        self, value, limit=..., inplace: bool = ..., downcast=...
    ) -> list[Block]: ...

class DatetimeLikeBlock(NDArrayBackedExtensionBlock):
    """Block for datetime64[ns], timedelta64[ns]."""

    __slots__ = ...
    is_numeric = ...
    values: DatetimeArray | TimedeltaArray

class DatetimeTZBlock(DatetimeLikeBlock):
    """implement a datetime64 block with a tz attribute"""

    values: DatetimeArray
    __slots__ = ...
    is_extension = ...
    _validate_ndim = ...
    _can_consolidate = ...

class ObjectBlock(NumpyBlock):
    __slots__ = ...
    is_object = ...
    @maybe_split
    def reduce(self, func, ignore_failures: bool = ...) -> list[Block]:
        """
        For object-dtype, we operate column-wise.
        """
        ...
    @maybe_split
    def convert(
        self,
        copy: bool = ...,
        datetime: bool = ...,
        numeric: bool = ...,
        timedelta: bool = ...,
    ) -> list[Block]:
        """
        attempt to cast any object types to better types return a copy of
        the block (if copy = True) by definition we ARE an ObjectBlock!!!!!
        """
        ...

class CategoricalBlock(ExtensionBlock):
    __slots__ = ...
    @property
    def dtype(self) -> DtypeObj: ...

def maybe_coerce_values(values) -> ArrayLike:
    """
    Input validation for values passed to __init__. Ensure that
    any datetime64/timedelta64 dtypes are in nanoseconds.  Ensure
    that we do not have string dtypes.

    Parameters
    ----------
    values : np.ndarray or ExtensionArray

    Returns
    -------
    values : np.ndarray or ExtensionArray
    """
    ...

def get_block_type(
    values, dtype: Dtype | None = ...
):  # -> Type[ExtensionBlock] | Type[CategoricalBlock] | Type[DatetimeTZBlock] | Type[DatetimeLikeBlock] | Type[NumericBlock] | Type[ObjectBlock]:
    """
    Find the appropriate Block subclass to use for the given values and dtype.

    Parameters
    ----------
    values : ndarray-like
    dtype : numpy or pandas dtype

    Returns
    -------
    cls : class, subclass of Block
    """
    ...

def new_block(values, placement, *, ndim: int, klass=...) -> Block: ...
def check_ndim(values, placement: BlockPlacement, ndim: int):  # -> None:
    """
    ndim inference and validation.

    Validates that values.ndim and ndim are consistent.
    Validates that len(values) and len(placement) are consistent.

    Parameters
    ----------
    values : array-like
    placement : BlockPlacement
    ndim : int

    Raises
    ------
    ValueError : the number of dimensions do not match
    """
    ...

def extract_pandas_array(
    values: np.ndarray | ExtensionArray, dtype: DtypeObj | None, ndim: int
) -> tuple[np.ndarray | ExtensionArray, DtypeObj | None]:
    """
    Ensure that we don't allow PandasArray / PandasDtype in internals.
    """
    ...

def extend_blocks(result, blocks=...) -> list[Block]:
    """return a new extended blocks, given the result"""
    ...

def ensure_block_shape(values: ArrayLike, ndim: int = ...) -> ArrayLike:
    """
    Reshape if possible to have values.ndim == ndim.
    """
    ...

def to_native_types(
    values: ArrayLike, *, na_rep=..., quoting=..., float_format=..., decimal=..., **kwargs
) -> np.ndarray:
    """convert to our native types format"""
    ...

def external_values(values: ArrayLike) -> ArrayLike:
    """
    The array that Series.values returns (public attribute).

    This has some historical constraints, and is overridden in block
    subclasses to return the correct array (e.g. period returns
    object ndarray and datetimetz a datetime64[ns] ndarray instead of
    proper extension array).
    """
    ...
