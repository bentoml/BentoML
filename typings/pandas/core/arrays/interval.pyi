import textwrap
from typing import Sequence
import numpy as np
from pandas._libs.interval import Interval, IntervalMixin
from pandas._typing import Dtype, NpDtype
from pandas.core.arrays.base import ExtensionArray, _extension_array_shared_docs
from pandas.core.dtypes.dtypes import IntervalDtype
from pandas.core.ops import unpack_zerodim_and_defer
from pandas.util._decorators import Appender

IntervalArrayT = ...
_interval_shared_docs: dict[str, str] = ...
_shared_docs_kwargs = ...

@Appender(
    _interval_shared_docs["class"]
    % {
        "klass": "IntervalArray",
        "summary": "Pandas array for interval data that are closed on the same side.",
        "versionadded": "0.24.0",
        "name": "",
        "extra_attributes": "",
        "extra_methods": "",
        "examples": textwrap.dedent(
            """    Examples
    --------
    A new ``IntervalArray`` can be constructed directly from an array-like of
    ``Interval`` objects:
    >>> pd.arrays.IntervalArray([pd.Interval(0, 1), pd.Interval(1, 5)])
    <IntervalArray>
    [(0, 1], (1, 5]]
    Length: 2, dtype: interval[int64, right]
    It may also be constructed using one of the constructor
    methods: :meth:`IntervalArray.from_arrays`,
    :meth:`IntervalArray.from_breaks`, and :meth:`IntervalArray.from_tuples`.
    """
        ),
    }
)
class IntervalArray(IntervalMixin, ExtensionArray):
    ndim = ...
    can_hold_na = ...
    _na_value = ...
    def __new__(
        cls: type[IntervalArrayT],
        data,
        closed=...,
        dtype: Dtype | None = ...,
        copy: bool = ...,
        verify_integrity: bool = ...,
    ): ...
    @classmethod
    @Appender(
        _interval_shared_docs["from_breaks"]
        % {
            "klass": "IntervalArray",
            "examples": textwrap.dedent(
                """        Examples
        --------
        >>> pd.arrays.IntervalArray.from_breaks([0, 1, 2, 3])
        <IntervalArray>
        [(0, 1], (1, 2], (2, 3]]
        Length: 3, dtype: interval[int64, right]
        """
            ),
        }
    )
    def from_breaks(
        cls: type[IntervalArrayT],
        breaks,
        closed=...,
        copy: bool = ...,
        dtype: Dtype | None = ...,
    ) -> IntervalArrayT: ...
    @classmethod
    @Appender(
        _interval_shared_docs["from_arrays"]
        % {
            "klass": "IntervalArray",
            "examples": textwrap.dedent(
                """        >>> pd.arrays.IntervalArray.from_arrays([0, 1, 2], [1, 2, 3])
        <IntervalArray>
        [(0, 1], (1, 2], (2, 3]]
        Length: 3, dtype: interval[int64, right]
        """
            ),
        }
    )
    def from_arrays(
        cls: type[IntervalArrayT],
        left,
        right,
        closed=...,
        copy: bool = ...,
        dtype: Dtype | None = ...,
    ) -> IntervalArrayT: ...
    @classmethod
    @Appender(
        _interval_shared_docs["from_tuples"]
        % {
            "klass": "IntervalArray",
            "examples": textwrap.dedent(
                """        Examples
        --------
        >>> pd.arrays.IntervalArray.from_tuples([(0, 1), (1, 2)])
        <IntervalArray>
        [(0, 1], (1, 2]]
        Length: 2, dtype: interval[int64, right]
        """
            ),
        }
    )
    def from_tuples(
        cls: type[IntervalArrayT],
        data,
        closed=...,
        copy: bool = ...,
        dtype: Dtype | None = ...,
    ) -> IntervalArrayT: ...
    @property
    def dtype(self) -> IntervalDtype: ...
    @property
    def nbytes(self) -> int: ...
    @property
    def size(self) -> int: ...
    def __iter__(self): ...
    def __len__(self) -> int: ...
    def __getitem__(self, key): ...
    def __setitem__(self, key, value): ...
    @unpack_zerodim_and_defer("__eq__")
    def __eq__(self, other) -> bool: ...
    @unpack_zerodim_and_defer("__ne__")
    def __ne__(self, other) -> bool: ...
    @unpack_zerodim_and_defer("__gt__")
    def __gt__(self, other) -> bool: ...
    @unpack_zerodim_and_defer("__ge__")
    def __ge__(self, other) -> bool: ...
    @unpack_zerodim_and_defer("__lt__")
    def __lt__(self, other) -> bool: ...
    @unpack_zerodim_and_defer("__le__")
    def __le__(self, other) -> bool: ...
    def argsort(
        self,
        ascending: bool = ...,
        kind: str = ...,
        na_position: str = ...,
        *args,
        **kwargs
    ) -> np.ndarray: ...
    def fillna(
        self: IntervalArrayT, value=..., method=..., limit=...
    ) -> IntervalArrayT: ...
    def astype(self, dtype, copy: bool = ...): ...
    def equals(self, other) -> bool: ...
    def copy(self: IntervalArrayT) -> IntervalArrayT: ...
    def isna(self) -> np.ndarray: ...
    def shift(
        self: IntervalArrayT, periods: int = ..., fill_value: object = ...
    ) -> IntervalArray: ...
    def take(
        self: IntervalArrayT,
        indices,
        *,
        allow_fill: bool = ...,
        fill_value=...,
        axis=...,
        **kwargs
    ) -> IntervalArrayT: ...
    def value_counts(self, dropna: bool = ...): ...
    def __repr__(self) -> str: ...
    @property
    def left(self): ...
    @property
    def right(self): ...
    @property
    def length(self): ...
    @property
    def mid(self): ...
    @Appender(
        _interval_shared_docs["overlaps"]
        % {
            "klass": "IntervalArray",
            "examples": textwrap.dedent(
                """        >>> data = [(0, 1), (1, 3), (2, 4)]
        >>> intervals = pd.arrays.IntervalArray.from_tuples(data)
        >>> intervals
        <IntervalArray>
        [(0, 1], (1, 3], (2, 4]]
        Length: 3, dtype: interval[int64, right]
        """
            ),
        }
    )
    def overlaps(self, other): ...
    @property
    def closed(self): ...
    @Appender(
        _interval_shared_docs["set_closed"]
        % {
            "klass": "IntervalArray",
            "examples": textwrap.dedent(
                """        Examples
        --------
        >>> index = pd.arrays.IntervalArray.from_breaks(range(4))
        >>> index
        <IntervalArray>
        [(0, 1], (1, 2], (2, 3]]
        Length: 3, dtype: interval[int64, right]
        >>> index.set_closed('both')
        <IntervalArray>
        [[0, 1], [1, 2], [2, 3]]
        Length: 3, dtype: interval[int64, both]
        """
            ),
        }
    )
    def set_closed(self: IntervalArrayT, closed) -> IntervalArrayT: ...
    @property
    @Appender(
        _interval_shared_docs["is_non_overlapping_monotonic"] % _shared_docs_kwargs
    )
    def is_non_overlapping_monotonic(self) -> bool: ...
    def __array__(self, dtype: NpDtype | None = ...) -> np.ndarray: ...
    def __arrow_array__(self, type=...): ...
    @Appender(
        _interval_shared_docs["to_tuples"] % {"return_type": "ndarray", "examples": ""}
    )
    def to_tuples(self, na_tuple=...) -> np.ndarray: ...
    def putmask(self, mask: np.ndarray, value) -> None: ...
    def insert(self: IntervalArrayT, loc: int, item: Interval) -> IntervalArrayT: ...
    def delete(self: IntervalArrayT, loc) -> IntervalArrayT: ...
    @Appender(_extension_array_shared_docs["repeat"] % _shared_docs_kwargs)
    def repeat(
        self: IntervalArrayT, repeats: int | Sequence[int], axis: int | None = ...
    ) -> IntervalArrayT: ...
    @Appender(
        _interval_shared_docs["contains"]
        % {
            "klass": "IntervalArray",
            "examples": textwrap.dedent(
                """        >>> intervals = pd.arrays.IntervalArray.from_tuples([(0, 1), (1, 3), (2, 4)])
        >>> intervals
        <IntervalArray>
        [(0, 1], (1, 3], (2, 4]]
        Length: 3, dtype: interval[int64, right]
        """
            ),
        }
    )
    def contains(self, other): ...
    def isin(self, values) -> np.ndarray: ...
    def unique(self) -> IntervalArray: ...
