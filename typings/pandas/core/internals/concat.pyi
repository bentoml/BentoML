from typing import TYPE_CHECKING

from pandas import Index
from pandas._typing import ArrayLike, DtypeObj, Manager, Shape
from pandas.util._decorators import cache_readonly

if TYPE_CHECKING: ...

def concat_arrays(to_concat: list) -> ArrayLike:
    """
    Alternative for concat_compat but specialized for use in the ArrayManager.

    Differences: only deals with 1D arrays (no axis keyword), assumes
    ensure_wrapped_if_datetimelike and does not skip empty arrays to determine
    the dtype.
    In addition ensures that all NullArrayProxies get replaced with actual
    arrays.

    Parameters
    ----------
    to_concat : list of arrays

    Returns
    -------
    np.ndarray or ExtensionArray
    """
    ...

def concatenate_managers(
    mgrs_indexers, axes: list[Index], concat_axis: int, copy: bool
) -> Manager:
    """
    Concatenate block managers into one.

    Parameters
    ----------
    mgrs_indexers : list of (BlockManager, {axis: indexer,...}) tuples
    axes : list of Index
    concat_axis : int
    copy : bool

    Returns
    -------
    BlockManager
    """
    ...

class JoinUnit:
    def __init__(self, block, shape: Shape, indexers=...) -> None: ...
    def __repr__(self) -> str: ...
    @cache_readonly
    def needs_filling(self) -> bool: ...
    @cache_readonly
    def dtype(self): ...
    def is_valid_na_for(self, dtype: DtypeObj) -> bool:
        """
        Check that we are all-NA of a type/dtype that is compatible with this dtype.
        Augments `self.is_na` with an additional check of the type of NA values.
        """
        ...
    @cache_readonly
    def is_na(self) -> bool: ...
    def get_reindexed_values(self, empty_dtype: DtypeObj, upcasted_na) -> ArrayLike: ...
