from pandas._typing import DtypeObj, Shape, final
from pandas.core.base import PandasObject
from pandas.core.indexes.api import Index

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
    ) -> T: ...
    def equals(self, other: object) -> bool: ...
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
    def array(self): ...

def interleaved_dtype(dtypes: list[DtypeObj]) -> DtypeObj | None: ...
