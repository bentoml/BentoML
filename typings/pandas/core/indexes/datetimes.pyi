from datetime import tzinfo
from typing import TYPE_CHECKING, Hashable
import numpy as np
from pandas import DataFrame, Float64Index, PeriodIndex, TimedeltaIndex
from pandas._libs import index as libindex
from pandas._typing import Dtype
from pandas.core.arrays.datetimes import DatetimeArray
from pandas.core.indexes.base import Index
from pandas.core.indexes.datetimelike import DatetimeTimedeltaMixin
from pandas.core.indexes.extension import inherit_names
from pandas.util._decorators import doc

if TYPE_CHECKING: ...

@inherit_names(
    DatetimeArray._field_ops
    + [
        method
        for method in DatetimeArray._datetimelike_methods
        if method not in ("tz_localize", "tz_convert")
    ],
    DatetimeArray,
    wrap=True,
)
@inherit_names(["is_normalized", "_resolution_obj"], DatetimeArray, cache=True)
@inherit_names(
    [
        "_bool_ops",
        "_object_ops",
        "_field_ops",
        "_datetimelike_ops",
        "_datetimelike_methods",
        "tz",
        "tzinfo",
        "dtype",
        "to_pydatetime",
        "_has_same_tz",
        "_format_native_types",
        "date",
        "time",
        "timetz",
        "std",
    ]
    + DatetimeArray._bool_ops,
    DatetimeArray,
)
class DatetimeIndex(DatetimeTimedeltaMixin):
    _typ = ...
    _data_cls = DatetimeArray
    _engine_type = libindex.DatetimeEngine
    _supports_partial_string_indexing = ...
    _data: DatetimeArray
    inferred_freq: str | None
    tz: tzinfo | None
    @doc(DatetimeArray.strftime)
    def strftime(self, date_format) -> Index: ...
    @doc(DatetimeArray.tz_convert)
    def tz_convert(self, tz) -> DatetimeIndex: ...
    @doc(DatetimeArray.tz_localize)
    def tz_localize(self, tz, ambiguous=..., nonexistent=...) -> DatetimeIndex: ...
    @doc(DatetimeArray.to_period)
    def to_period(self, freq=...) -> PeriodIndex: ...
    @doc(DatetimeArray.to_perioddelta)
    def to_perioddelta(self, freq) -> TimedeltaIndex: ...
    @doc(DatetimeArray.to_julian_date)
    def to_julian_date(self) -> Float64Index: ...
    @doc(DatetimeArray.isocalendar)
    def isocalendar(self) -> DataFrame: ...
    def __new__(
        cls,
        data=...,
        freq=...,
        tz=...,
        normalize: bool = ...,
        closed=...,
        ambiguous=...,
        dayfirst: bool = ...,
        yearfirst: bool = ...,
        dtype: Dtype | None = ...,
        copy: bool = ...,
        name: Hashable = ...,
    ) -> DatetimeIndex: ...
    def __reduce__(self): ...
    def union_many(self, others): ...
    def to_series(self, keep_tz=..., index=..., name=...): ...
    def snap(self, freq=...) -> DatetimeIndex: ...
    def get_loc(self, key, method=..., tolerance=...): ...
    def slice_indexer(self, start=..., end=..., step=..., kind=...): ...
    @property
    def inferred_type(self) -> str: ...
    def indexer_at_time(self, time, asof: bool = ...) -> np.ndarray: ...
    def indexer_between_time(
        self, start_time, end_time, include_start: bool = ..., include_end: bool = ...
    ) -> np.ndarray: ...

def date_range(
    start=...,
    end=...,
    periods=...,
    freq=...,
    tz=...,
    normalize: bool = ...,
    name: Hashable = ...,
    closed=...,
    **kwargs
) -> DatetimeIndex: ...
def bdate_range(
    start=...,
    end=...,
    periods: int | None = ...,
    freq=...,
    tz=...,
    normalize: bool = ...,
    name: Hashable = ...,
    weekmask=...,
    holidays=...,
    closed=...,
    **kwargs
) -> DatetimeIndex: ...
