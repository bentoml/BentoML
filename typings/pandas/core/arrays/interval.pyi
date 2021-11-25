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
            """\
    Examples
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
                """\
        Examples
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
                """\
        >>> pd.arrays.IntervalArray.from_arrays([0, 1, 2], [1, 2, 3])
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
                """\
        Examples
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
    def fillna(self: IntervalArrayT, value=..., method=..., limit=...) -> IntervalArrayT:
        """
        Fill NA/NaN values using the specified method.

        Parameters
        ----------
        value : scalar, dict, Series
            If a scalar value is passed it is used to fill all missing values.
            Alternatively, a Series or dict can be used to fill in different
            values for each index. The value should not be a list. The
            value(s) passed should be either Interval objects or NA/NaN.
        method : {'backfill', 'bfill', 'pad', 'ffill', None}, default None
            (Not implemented yet for IntervalArray)
            Method to use for filling holes in reindexed Series
        limit : int, default None
            (Not implemented yet for IntervalArray)
            If method is specified, this is the maximum number of consecutive
            NaN values to forward/backward fill. In other words, if there is
            a gap with more than this number of consecutive NaNs, it will only
            be partially filled. If method is not specified, this is the
            maximum number of entries along the entire axis where NaNs will be
            filled.

        Returns
        -------
        filled : IntervalArray with NA/NaN filled
        """
        ...
    def astype(
        self, dtype, copy: bool = ...
    ):  # -> Self@IntervalArray | Categorical | NDArray:
        """
        Cast to an ExtensionArray or NumPy array with dtype 'dtype'.

        Parameters
        ----------
        dtype : str or dtype
            Typecode or data-type to which the array is cast.

        copy : bool, default True
            Whether to copy the data, even if not necessary. If False,
            a copy is made only if the old dtype does not match the
            new dtype.

        Returns
        -------
        array : ExtensionArray or ndarray
            ExtensionArray or NumPy ndarray with 'dtype' for its dtype.
        """
        ...
    def equals(self, other) -> bool: ...
    def copy(self: IntervalArrayT) -> IntervalArrayT:
        """
        Return a copy of the array.

        Returns
        -------
        IntervalArray
        """
        ...
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
    ) -> IntervalArrayT:
        """
        Take elements from the IntervalArray.

        Parameters
        ----------
        indices : sequence of integers
            Indices to be taken.

        allow_fill : bool, default False
            How to handle negative values in `indices`.

            * False: negative values in `indices` indicate positional indices
              from the right (the default). This is similar to
              :func:`numpy.take`.

            * True: negative values in `indices` indicate
              missing values. These values are set to `fill_value`. Any other
              other negative values raise a ``ValueError``.

        fill_value : Interval or NA, optional
            Fill value to use for NA-indices when `allow_fill` is True.
            This may be ``None``, in which case the default NA value for
            the type, ``self.dtype.na_value``, is used.

            For many ExtensionArrays, there will be two representations of
            `fill_value`: a user-facing "boxed" scalar, and a low-level
            physical NA value. `fill_value` should be the user-facing version,
            and the implementation should handle translating that to the
            physical version for processing the take if necessary.

        axis : any, default None
            Present for compat with IntervalIndex; does nothing.

        Returns
        -------
        IntervalArray

        Raises
        ------
        IndexError
            When the indices are out of bounds for the array.
        ValueError
            When `indices` contains negative values other than ``-1``
            and `allow_fill` is True.
        """
        ...
    def value_counts(self, dropna: bool = ...):  # -> Series:
        """
        Returns a Series containing counts of each interval.

        Parameters
        ----------
        dropna : bool, default True
            Don't include counts of NaN.

        Returns
        -------
        counts : Series

        See Also
        --------
        Series.value_counts
        """
        ...
    def __repr__(self) -> str: ...
    @property
    def left(self):  # -> Index:
        """
        Return the left endpoints of each Interval in the IntervalArray as
        an Index.
        """
        ...
    @property
    def right(self):  # -> Index:
        """
        Return the right endpoints of each Interval in the IntervalArray as
        an Index.
        """
        ...
    @property
    def length(self):  # -> _NotImplementedType:
        """
        Return an Index with entries denoting the length of each Interval in
        the IntervalArray.
        """
        ...
    @property
    def mid(self):  # -> float | _NotImplementedType:
        """
        Return the midpoint of each Interval in the IntervalArray as an Index.
        """
        ...
    @Appender(
        _interval_shared_docs["overlaps"]
        % {
            "klass": "IntervalArray",
            "examples": textwrap.dedent(
                """\
        >>> data = [(0, 1), (1, 3), (2, 4)]
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
    def closed(self):
        """
        Whether the intervals are closed on the left-side, right-side, both or
        neither.
        """
        ...
    @Appender(
        _interval_shared_docs["set_closed"]
        % {
            "klass": "IntervalArray",
            "examples": textwrap.dedent(
                """\
        Examples
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
    @Appender(_interval_shared_docs["is_non_overlapping_monotonic"] % _shared_docs_kwargs)
    def is_non_overlapping_monotonic(self) -> bool: ...
    def __array__(self, dtype: NpDtype | None = ...) -> np.ndarray:
        """
        Return the IntervalArray's data as a numpy array of Interval
        objects (with dtype='object')
        """
        ...
    def __arrow_array__(self, type=...):
        """
        Convert myself into a pyarrow Array.
        """
        ...
    @Appender(
        _interval_shared_docs["to_tuples"] % {"return_type": "ndarray", "examples": ""}
    )
    def to_tuples(self, na_tuple=...) -> np.ndarray: ...
    def putmask(self, mask: np.ndarray, value) -> None: ...
    def insert(self: IntervalArrayT, loc: int, item: Interval) -> IntervalArrayT:
        """
        Return a new IntervalArray inserting new item at location. Follows
        Python list.append semantics for negative values.  Only Interval
        objects and NA can be inserted into an IntervalIndex

        Parameters
        ----------
        loc : int
        item : Interval

        Returns
        -------
        IntervalArray
        """
        ...
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
                """\
        >>> intervals = pd.arrays.IntervalArray.from_tuples([(0, 1), (1, 3), (2, 4)])
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
