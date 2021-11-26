from typing import Hashable
import numpy as np
from pandas._libs import index as libindex
from pandas._libs.tslibs import BaseOffset
from pandas._typing import Dtype
from pandas.core.arrays.period import PeriodArray
from pandas.core.indexes.datetimelike import DatetimeIndexOpsMixin
from pandas.core.indexes.datetimes import DatetimeIndex, Index
from pandas.core.indexes.extension import inherit_names
from pandas.core.indexes.numeric import Int64Index
from pandas.util._decorators import doc

_index_doc_kwargs = ...
_shared_doc_kwargs = ...

@inherit_names(
    ["strftime", "start_time", "end_time"] + PeriodArray._field_ops,
    PeriodArray,
    wrap=True,
)
@inherit_names(["is_leap_year", "_format_native_types"], PeriodArray)
class PeriodIndex(DatetimeIndexOpsMixin):
    _typ = ...
    _attributes = ...
    _data: PeriodArray
    freq: BaseOffset
    _data_cls = PeriodArray
    _engine_type = libindex.PeriodEngine
    _supports_partial_string_indexing = ...
    @doc(
        PeriodArray.asfreq,
        other="pandas.arrays.PeriodArray",
        other_name="PeriodArray",
        **_shared_doc_kwargs
    )
    def asfreq(self, freq=..., how: str = ...) -> PeriodIndex: ...
    @doc(PeriodArray.to_timestamp)
    def to_timestamp(self, freq=..., how=...) -> DatetimeIndex: ...
    @property
    @doc(PeriodArray.hour.fget)
    def hour(self) -> Int64Index: ...
    @property
    @doc(PeriodArray.minute.fget)
    def minute(self) -> Int64Index: ...
    @property
    @doc(PeriodArray.second.fget)
    def second(self) -> Int64Index: ...
    def __new__(
        cls,
        data=...,
        ordinal=...,
        freq=...,
        dtype: Dtype | None = ...,
        copy: bool = ...,
        name: Hashable = ...,
        **fields
    ) -> PeriodIndex: ...
    @property
    def values(self) -> np.ndarray: ...
    def asof_locs(self, where: Index, mask: np.ndarray) -> np.ndarray: ...
    @doc(Index.astype)
    def astype(self, dtype, copy: bool = ..., how=...): ...
    @property
    def is_full(self) -> bool: ...
    @property
    def inferred_type(self) -> str: ...
    def get_loc(self, key, method=..., tolerance=...): ...

def period_range(
    start=..., end=..., periods: int | None = ..., freq=..., name=...
) -> PeriodIndex: ...
