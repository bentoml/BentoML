from typing import Any, Callable, Hashable, Sequence
import numpy as np
from pandas._libs import internals as libinternals
from pandas._typing import ArrayLike, Dtype, DtypeObj, Shape, type_t
from pandas.core.indexes.api import Float64Index, Index
from pandas.core.internals.base import DataManager, SingleDataManager
from pandas.core.internals.blocks import Block

T = ...

class BaseBlockManager(DataManager):
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
    def blknos(self): ...
    @property
    def blklocs(self): ...
    def make_empty(self: T, axes=...) -> T: ...
    def __nonzero__(self) -> bool: ...
    __bool__ = ...
    def set_axis(self, axis: int, new_labels: Index) -> None: ...
    @property
    def is_single_block(self) -> bool: ...
    @property
    def items(self) -> Index: ...
    def get_dtypes(self): ...
    @property
    def arrays(self) -> list[ArrayLike]: ...
    def __repr__(self) -> str: ...
    def apply(
        self: T,
        f,
        align_keys: list[str] | None = ...,
        ignore_failures: bool = ...,
        **kwargs
    ) -> T: ...
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
    ) -> T: ...
    def to_native_types(self: T, **kwargs) -> T: ...
    def is_consolidated(self) -> bool: ...
    @property
    def is_numeric_mixed_type(self) -> bool: ...
    @property
    def any_extension_types(self) -> bool: ...
    @property
    def is_view(self) -> bool: ...
    def get_bool_data(self: T, copy: bool = ...) -> T: ...
    def get_numeric_data(self: T, copy: bool = ...) -> T: ...
    @property
    def nblocks(self) -> int: ...
    def copy(self: T, deep=...) -> T: ...
    def consolidate(self: T) -> T: ...
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
    ) -> T: ...
    def take(self: T, indexer, axis: int = ..., verify: bool = ...) -> T: ...

class BlockManager(libinternals.BlockManager, BaseBlockManager):
    ndim = ...
    def __init__(
        self,
        blocks: Sequence[Block],
        axes: Sequence[Index],
        verify_integrity: bool = ...,
    ) -> None: ...
    @classmethod
    def from_blocks(cls, blocks: list[Block], axes: list[Index]) -> BlockManager: ...
    def fast_xs(self, loc: int) -> ArrayLike: ...
    def iget(self, i: int) -> SingleBlockManager: ...
    def iget_values(self, i: int) -> ArrayLike: ...
    @property
    def column_arrays(self) -> list[np.ndarray]: ...
    def iset(self, loc: int | slice | np.ndarray, value: ArrayLike): ...
    def insert(self, loc: int, item: Hashable, value: ArrayLike) -> None: ...
    def idelete(self, indexer) -> BlockManager: ...
    def grouped_reduce(self: T, func: Callable, ignore_failures: bool = ...) -> T: ...
    def reduce(
        self: T, func: Callable, ignore_failures: bool = ...
    ) -> tuple[T, np.ndarray]: ...
    def operate_blockwise(self, other: BlockManager, array_op) -> BlockManager: ...
    def quantile(
        self: T, *, qs: Float64Index, axis: int = ..., interpolation=...
    ) -> T: ...
    def unstack(self, unstacker, fill_value) -> BlockManager: ...
    def to_dict(self, copy: bool = ...): ...
    def as_array(
        self,
        transpose: bool = ...,
        dtype: Dtype | None = ...,
        copy: bool = ...,
        na_value=...,
    ) -> np.ndarray: ...

class SingleBlockManager(BaseBlockManager, SingleDataManager):
    ndim = ...
    _is_consolidated = ...
    _known_consolidated = ...
    __slots__ = ...
    is_single_block = ...
    def __init__(
        self, block: Block, axis: Index, verify_integrity: bool = ..., fastpath=...
    ) -> None: ...
    @classmethod
    def from_blocks(
        cls, blocks: list[Block], axes: list[Index]
    ) -> SingleBlockManager: ...
    @classmethod
    def from_array(cls, array: ArrayLike, index: Index) -> SingleBlockManager: ...
    def __getstate__(self): ...
    def __setstate__(self, state): ...
    def getitem_mgr(self, indexer) -> SingleBlockManager: ...
    def get_slice(self, slobj: slice, axis: int = ...) -> SingleBlockManager: ...
    @property
    def index(self) -> Index: ...
    @property
    def dtype(self) -> DtypeObj: ...
    def get_dtypes(self) -> np.ndarray: ...
    def external_values(self): ...
    def internal_values(self): ...
    def array_values(self): ...
    def is_consolidated(self) -> bool: ...
    def idelete(self, indexer) -> SingleBlockManager: ...
    def fast_xs(self, loc): ...
    def set_values(self, values: ArrayLike): ...

def create_block_manager_from_blocks(
    blocks: list[Block], axes: list[Index], consolidate: bool = ...
) -> BlockManager: ...
def create_block_manager_from_arrays(
    arrays, names: Index, axes: list[Index], consolidate: bool = ...
) -> BlockManager: ...
