from pandas.core.indexes.base import Index

_sort_msg = ...
__all__ = [
    "Index",
    "MultiIndex",
    "NumericIndex",
    "Float64Index",
    "Int64Index",
    "CategoricalIndex",
    "IntervalIndex",
    "RangeIndex",
    "UInt64Index",
    "InvalidIndexError",
    "TimedeltaIndex",
    "PeriodIndex",
    "DatetimeIndex",
    "_new_Index",
    "NaT",
    "ensure_index",
    "ensure_index_from_sequences",
    "get_objs_combined_axis",
    "union_indexes",
    "get_unanimous_names",
    "all_indexes_same",
]

def get_objs_combined_axis(
    objs, intersect: bool = ..., axis=..., sort: bool = ..., copy: bool = ...
) -> Index: ...
def union_indexes(indexes, sort: bool = ...) -> Index: ...
def all_indexes_same(indexes) -> bool: ...
