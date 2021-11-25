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
) -> Index:
    """
    Extract combined index: return intersection or union (depending on the
    value of "intersect") of indexes on given axis, or None if all objects
    lack indexes (e.g. they are numpy arrays).

    Parameters
    ----------
    objs : list
        Series or DataFrame objects, may be mix of the two.
    intersect : bool, default False
        If True, calculate the intersection between indexes. Otherwise,
        calculate the union.
    axis : {0 or 'index', 1 or 'outer'}, default 0
        The axis to extract indexes from.
    sort : bool, default True
        Whether the result index should come out sorted or not.
    copy : bool, default False
        If True, return a copy of the combined index.

    Returns
    -------
    Index
    """
    ...

def union_indexes(indexes, sort: bool = ...) -> Index:
    """
    Return the union of indexes.

    The behavior of sort and names is not consistent.

    Parameters
    ----------
    indexes : list of Index or list objects
    sort : bool, default True
        Whether the result index should come out sorted or not.

    Returns
    -------
    Index
    """
    ...

def all_indexes_same(indexes) -> bool:
    """
    Determine if all indexes contain the same elements.

    Parameters
    ----------
    indexes : iterable of Index objects

    Returns
    -------
    bool
        True if all indexes contain the same elements, False otherwise.
    """
    ...
