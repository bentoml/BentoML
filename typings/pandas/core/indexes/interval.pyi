import textwrap
from typing import Any, Hashable
import numpy as np
from pandas._typing import Dtype
from pandas.core.arrays.interval import IntervalArray, _interval_shared_docs
from pandas.core.indexes.base import Index, _index_shared_docs
from pandas.core.indexes.extension import ExtensionIndex, inherit_names
from pandas.util._decorators import Appender, cache_readonly

_index_doc_kwargs = ...

@Appender(
    _interval_shared_docs["class"]
    % {
        "klass": "IntervalIndex",
        "summary": "Immutable index of intervals that are closed on the same side.",
        "name": _index_doc_kwargs["name"],
        "versionadded": "0.20.0",
        "extra_attributes": "is_overlapping\nvalues\n",
        "extra_methods": "",
        "examples": textwrap.dedent(
            """    Examples
    --------
    A new ``IntervalIndex`` is typically constructed using
    :func:`interval_range`:
    >>> pd.interval_range(start=0, end=5)
    IntervalIndex([(0, 1], (1, 2], (2, 3], (3, 4], (4, 5]],
                  dtype='interval[int64, right]')
    It may also be constructed using one of the constructor
    methods: :meth:`IntervalIndex.from_arrays`,
    :meth:`IntervalIndex.from_breaks`, and :meth:`IntervalIndex.from_tuples`.
    See further examples in the doc strings of ``interval_range`` and the
    mentioned constructor methods.
    """
        ),
    }
)
@inherit_names(["set_closed", "to_tuples"], IntervalArray, wrap=True)
@inherit_names(
    [
        "__array__",
        "overlaps",
        "contains",
        "closed_left",
        "closed_right",
        "open_left",
        "open_right",
        "is_empty",
    ],
    IntervalArray,
)
@inherit_names(["is_non_overlapping_monotonic", "closed"], IntervalArray, cache=True)
class IntervalIndex(ExtensionIndex):
    _typ = ...
    closed: str
    is_non_overlapping_monotonic: bool
    closed_left: bool
    closed_right: bool
    _data: IntervalArray
    _values: IntervalArray
    _can_hold_strings = ...
    _data_cls = IntervalArray
    def __new__(
        cls,
        data,
        closed=...,
        dtype: Dtype | None = ...,
        copy: bool = ...,
        name: Hashable = ...,
        verify_integrity: bool = ...,
    ) -> IntervalIndex: ...
    @classmethod
    @Appender(
        _interval_shared_docs["from_breaks"]
        % {
            "klass": "IntervalIndex",
            "examples": textwrap.dedent(
                """        Examples
        --------
        >>> pd.IntervalIndex.from_breaks([0, 1, 2, 3])
        IntervalIndex([(0, 1], (1, 2], (2, 3]],
                      dtype='interval[int64, right]')
        """
            ),
        }
    )
    def from_breaks(
        cls,
        breaks,
        closed: str = ...,
        name: Hashable = ...,
        copy: bool = ...,
        dtype: Dtype | None = ...,
    ) -> IntervalIndex: ...
    @classmethod
    @Appender(
        _interval_shared_docs["from_arrays"]
        % {
            "klass": "IntervalIndex",
            "examples": textwrap.dedent(
                """        Examples
        --------
        >>> pd.IntervalIndex.from_arrays([0, 1, 2], [1, 2, 3])
        IntervalIndex([(0, 1], (1, 2], (2, 3]],
                      dtype='interval[int64, right]')
        """
            ),
        }
    )
    def from_arrays(
        cls,
        left,
        right,
        closed: str = ...,
        name: Hashable = ...,
        copy: bool = ...,
        dtype: Dtype | None = ...,
    ) -> IntervalIndex: ...
    @classmethod
    @Appender(
        _interval_shared_docs["from_tuples"]
        % {
            "klass": "IntervalIndex",
            "examples": textwrap.dedent(
                """        Examples
        --------
        >>> pd.IntervalIndex.from_tuples([(0, 1), (1, 2)])
        IntervalIndex([(0, 1], (1, 2]],
                       dtype='interval[int64, right]')
        """
            ),
        }
    )
    def from_tuples(
        cls,
        data,
        closed: str = ...,
        name: Hashable = ...,
        copy: bool = ...,
        dtype: Dtype | None = ...,
    ) -> IntervalIndex: ...
    def __contains__(self, key: Any) -> bool: ...
    def __reduce__(self): ...
    @property
    def inferred_type(self) -> str: ...
    @Appender(Index.memory_usage.__doc__)
    def memory_usage(self, deep: bool = ...) -> int: ...
    @cache_readonly
    def is_monotonic_decreasing(self) -> bool: ...
    @cache_readonly
    def is_unique(self) -> bool: ...
    @property
    def is_overlapping(self) -> bool: ...
    def get_loc(
        self, key, method: str | None = ..., tolerance=...
    ) -> int | slice | np.ndarray: ...
    @Appender(_index_shared_docs["get_indexer_non_unique"] % _index_doc_kwargs)
    def get_indexer_non_unique(
        self, target: Index
    ) -> tuple[np.ndarray, np.ndarray]: ...
    _requires_unique_msg = ...
    @cache_readonly
    def left(self) -> Index: ...
    @cache_readonly
    def right(self) -> Index: ...
    @cache_readonly
    def mid(self) -> Index: ...
    @property
    def length(self) -> Index: ...

def interval_range(
    start=..., end=..., periods=..., freq=..., name: Hashable = ..., closed=...
) -> IntervalIndex: ...
