from typing import TYPE_CHECKING, Hashable, Iterable, Mapping, overload
from pandas import DataFrame
from pandas._typing import FrameOrSeriesUnion
from pandas.core.generic import NDFrame
from pandas.util._decorators import deprecate_nonkeyword_arguments

if TYPE_CHECKING: ...

@overload
def concat(
    objs: Iterable[DataFrame] | Mapping[Hashable, DataFrame],
    axis=...,
    join: str = ...,
    ignore_index: bool = ...,
    keys=...,
    levels=...,
    names=...,
    verify_integrity: bool = ...,
    sort: bool = ...,
    copy: bool = ...,
) -> DataFrame: ...
@overload
def concat(
    objs: Iterable[NDFrame] | Mapping[Hashable, NDFrame],
    axis=...,
    join: str = ...,
    ignore_index: bool = ...,
    keys=...,
    levels=...,
    names=...,
    verify_integrity: bool = ...,
    sort: bool = ...,
    copy: bool = ...,
) -> FrameOrSeriesUnion: ...
@deprecate_nonkeyword_arguments(version=None, allowed_args=["objs"])
def concat(
    objs: Iterable[NDFrame] | Mapping[Hashable, NDFrame],
    axis=...,
    join=...,
    ignore_index: bool = ...,
    keys=...,
    levels=...,
    names=...,
    verify_integrity: bool = ...,
    sort: bool = ...,
    copy: bool = ...,
) -> FrameOrSeriesUnion: ...

class _Concatenator:
    def __init__(
        self,
        objs: Iterable[NDFrame] | Mapping[Hashable, NDFrame],
        axis=...,
        join: str = ...,
        keys=...,
        levels=...,
        names=...,
        ignore_index: bool = ...,
        verify_integrity: bool = ...,
        copy: bool = ...,
        sort=...,
    ) -> None: ...
