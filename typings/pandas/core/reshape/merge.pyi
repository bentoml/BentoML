from typing import TYPE_CHECKING, Hashable
import numpy as np
from pandas import DataFrame, Index, MultiIndex, Series
from pandas._typing import IndexLabel, Suffixes
from pandas.core.frame import _merge_doc
from pandas.util._decorators import Appender, Substitution

if TYPE_CHECKING: ...

@Substitution("\nleft : DataFrame or named Series")
@Appender(_merge_doc, indents=0)
def merge(
    left: DataFrame | Series,
    right: DataFrame | Series,
    how: str = ...,
    on: IndexLabel | None = ...,
    left_on: IndexLabel | None = ...,
    right_on: IndexLabel | None = ...,
    left_index: bool = ...,
    right_index: bool = ...,
    sort: bool = ...,
    suffixes: Suffixes = ...,
    copy: bool = ...,
    indicator: bool = ...,
    validate: str | None = ...,
) -> DataFrame: ...

if __debug__: ...

def merge_ordered(
    left: DataFrame,
    right: DataFrame,
    on: IndexLabel | None = ...,
    left_on: IndexLabel | None = ...,
    right_on: IndexLabel | None = ...,
    left_by=...,
    right_by=...,
    fill_method: str | None = ...,
    suffixes: Suffixes = ...,
    how: str = ...,
) -> DataFrame: ...
def merge_asof(
    left: DataFrame | Series,
    right: DataFrame | Series,
    on: IndexLabel | None = ...,
    left_on: IndexLabel | None = ...,
    right_on: IndexLabel | None = ...,
    left_index: bool = ...,
    right_index: bool = ...,
    by=...,
    left_by=...,
    right_by=...,
    suffixes: Suffixes = ...,
    tolerance=...,
    allow_exact_matches: bool = ...,
    direction: str = ...,
) -> DataFrame: ...

class _MergeOperation:
    _merge_type = ...
    def __init__(
        self,
        left: DataFrame | Series,
        right: DataFrame | Series,
        how: str = ...,
        on: IndexLabel | None = ...,
        left_on: IndexLabel | None = ...,
        right_on: IndexLabel | None = ...,
        axis: int = ...,
        left_index: bool = ...,
        right_index: bool = ...,
        sort: bool = ...,
        suffixes: Suffixes = ...,
        copy: bool = ...,
        indicator: bool = ...,
        validate: str | None = ...,
    ) -> None: ...
    def get_result(self) -> DataFrame: ...

def get_join_indexers(
    left_keys, right_keys, sort: bool = ..., how: str = ..., **kwargs
) -> tuple[np.ndarray, np.ndarray]: ...
def restore_dropped_levels_multijoin(
    left: MultiIndex,
    right: MultiIndex,
    dropped_level_names,
    join_index: Index,
    lindexer: np.ndarray,
    rindexer: np.ndarray,
) -> tuple[list[Index], np.ndarray, list[Hashable]]: ...

class _OrderedMerge(_MergeOperation):
    _merge_type = ...
    def __init__(
        self,
        left: DataFrame | Series,
        right: DataFrame | Series,
        on: IndexLabel | None = ...,
        left_on: IndexLabel | None = ...,
        right_on: IndexLabel | None = ...,
        left_index: bool = ...,
        right_index: bool = ...,
        axis: int = ...,
        suffixes: Suffixes = ...,
        copy: bool = ...,
        fill_method: str | None = ...,
        how: str = ...,
    ) -> None: ...
    def get_result(self) -> DataFrame: ...

_type_casters = ...

class _AsOfMerge(_OrderedMerge):
    _merge_type = ...
    def __init__(
        self,
        left: DataFrame | Series,
        right: DataFrame | Series,
        on: IndexLabel | None = ...,
        left_on: IndexLabel | None = ...,
        right_on: IndexLabel | None = ...,
        left_index: bool = ...,
        right_index: bool = ...,
        by=...,
        left_by=...,
        right_by=...,
        axis: int = ...,
        suffixes: Suffixes = ...,
        copy: bool = ...,
        fill_method: str | None = ...,
        how: str = ...,
        tolerance=...,
        allow_exact_matches: bool = ...,
        direction: str = ...,
    ) -> None: ...
