from typing import Hashable
import numpy as np
from pandas._typing import ArrayLike, FrameOrSeries, final
from pandas.core.arrays import Categorical
from pandas.core.groupby import ops
from pandas.core.indexes.api import Index
from pandas.util._decorators import cache_readonly

class Grouper:
    axis: int
    sort: bool
    dropna: bool
    _gpr_index: Index | None
    _grouper: Index | None
    _attributes: tuple[str, ...] = ...
    def __new__(cls, *args, **kwargs): ...
    def __init__(
        self,
        key=...,
        level=...,
        freq=...,
        axis: int = ...,
        sort: bool = ...,
        dropna: bool = ...,
    ) -> None: ...
    @final
    @property
    def ax(self) -> Index: ...
    @final
    @property
    def groups(self): ...
    @final
    def __repr__(self) -> str: ...

@final
class Grouping:
    _codes: np.ndarray | None = ...
    _group_index: Index | None = ...
    _ed_categorical: bool
    _all_grouper: Categorical | None
    _index: Index
    def __init__(
        self,
        index: Index,
        grouper=...,
        obj: FrameOrSeries | None = ...,
        level=...,
        sort: bool = ...,
        observed: bool = ...,
        in_axis: bool = ...,
        dropna: bool = ...,
    ) -> None: ...
    def __repr__(self) -> str: ...
    def __iter__(self): ...
    @cache_readonly
    def name(self) -> Hashable: ...
    @property
    def ngroups(self) -> int: ...
    @cache_readonly
    def indices(self): ...
    @property
    def codes(self) -> np.ndarray: ...
    @cache_readonly
    def group_arraylike(self) -> ArrayLike: ...
    @cache_readonly
    def result_index(self) -> Index: ...
    @cache_readonly
    def group_index(self) -> Index: ...
    @cache_readonly
    def groups(self) -> dict[Hashable, np.ndarray]: ...

def get_grouper(
    obj: FrameOrSeries,
    key=...,
    axis: int = ...,
    level=...,
    sort: bool = ...,
    observed: bool = ...,
    mutated: bool = ...,
    validate: bool = ...,
    dropna: bool = ...,
) -> tuple[ops.BaseGrouper, frozenset[Hashable], FrameOrSeries]: ...
