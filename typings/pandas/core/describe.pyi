from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Callable, Sequence
from pandas import DataFrame, Series
from pandas._typing import FrameOrSeries, FrameOrSeriesUnion, Hashable

if TYPE_CHECKING: ...

def describe_ndframe(
    *,
    obj: FrameOrSeries,
    include: str | Sequence[str] | None,
    exclude: str | Sequence[str] | None,
    datetime_is_numeric: bool,
    percentiles: Sequence[float] | None
) -> FrameOrSeries: ...

class NDFrameDescriberAbstract(ABC):
    def __init__(self, obj: FrameOrSeriesUnion, datetime_is_numeric: bool) -> None: ...
    @abstractmethod
    def describe(self, percentiles: Sequence[float]) -> FrameOrSeriesUnion: ...

class SeriesDescriber(NDFrameDescriberAbstract):
    obj: Series
    def describe(self, percentiles: Sequence[float]) -> Series: ...

class DataFrameDescriber(NDFrameDescriberAbstract):
    def __init__(
        self,
        obj: DataFrame,
        *,
        include: str | Sequence[str] | None,
        exclude: str | Sequence[str] | None,
        datetime_is_numeric: bool
    ) -> None: ...
    def describe(self, percentiles: Sequence[float]) -> DataFrame: ...

def reorder_columns(ldesc: Sequence[Series]) -> list[Hashable]: ...
def describe_numeric_1d(series: Series, percentiles: Sequence[float]) -> Series: ...
def describe_categorical_1d(
    data: Series, percentiles_ignored: Sequence[float]
) -> Series: ...
def describe_timestamp_as_categorical_1d(
    data: Series, percentiles_ignored: Sequence[float]
) -> Series: ...
def describe_timestamp_1d(data: Series, percentiles: Sequence[float]) -> Series: ...
def select_describe_func(data: Series, datetime_is_numeric: bool) -> Callable: ...
def refine_percentiles(percentiles: Sequence[float] | None) -> Sequence[float]: ...
