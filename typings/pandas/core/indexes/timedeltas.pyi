from pandas._libs import index as libindex
from pandas._typing import Optional
from pandas.core.arrays.timedeltas import TimedeltaArray
from pandas.core.indexes.datetimelike import DatetimeTimedeltaMixin
from pandas.core.indexes.extension import inherit_names

@inherit_names(
    ["__neg__", "__pos__", "__abs__", "total_seconds", "round", "floor", "ceil"]
    + TimedeltaArray._field_ops,
    TimedeltaArray,
    wrap=True,
)
@inherit_names(
    ["components", "to_pytimedelta", "sum", "std", "median", "_format_native_types"],
    TimedeltaArray,
)
class TimedeltaIndex(DatetimeTimedeltaMixin):
    _typ = ...
    _data_cls = TimedeltaArray
    _engine_type = libindex.TimedeltaEngine
    _data: TimedeltaArray
    def __new__(
        cls, data=..., unit=..., freq=..., closed=..., dtype=..., copy=..., name=...
    ): ...
    def get_loc(self, key, method=..., tolerance=...): ...
    @property
    def inferred_type(self) -> str: ...

def timedelta_range(
    start=..., end=..., periods: Optional[int] = ..., freq=..., name=..., closed=...
) -> TimedeltaIndex: ...
