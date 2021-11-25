from pandas._typing import DtypeObj, Shape, final
from pandas.core.base import PandasObject
from pandas.core.indexes.api import Index

"""
Base class for the internal managers. Both BlockManager and ArrayManager
inherit from this class.
"""
T = ...

class DataManager(PandasObject):
    axes: list[Index]
    @property
    def items(self) -> Index: ...
    def __len__(self) -> int: ...
    @property
    def ndim(self) -> int: ...
    @property
    def shape(self) -> Shape: ...
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
    ) -> T: ...
    @final
    def reindex_axis(
        self: T,
        new_index: Index,
        axis: int,
        fill_value=...,
        consolidate: bool = ...,
        only_slice: bool = ...,
    ) -> T:
        """
        Conform data manager to new index.
        """
        ...
    def equals(self, other: object) -> bool:
        """
        Implementation for DataFrame.equals
        """
        ...
    def apply(
        self: T,
        f,
        align_keys: list[str] | None = ...,
        ignore_failures: bool = ...,
        **kwargs
    ) -> T: ...
    def isna(self: T, func) -> T: ...

class SingleDataManager(DataManager):
    ndim = ...
    @property
    def array(self):
        """
        Quick access to the backing array of the Block or SingleArrayManager.
        """
        ...

def interleaved_dtype(dtypes: list[DtypeObj]) -> DtypeObj | None:
    """
    Find the common dtype for `blocks`.

    Parameters
    ----------
    blocks : List[DtypeObj]

    Returns
    -------
    dtype : np.dtype, ExtensionDtype, or None
        None is returned when `blocks` is empty.
    """
    ...
