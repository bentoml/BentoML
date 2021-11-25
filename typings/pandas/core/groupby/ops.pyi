from typing import Generic, Hashable, Iterator, Sequence

import numpy as np
from pandas._typing import ArrayLike, F, FrameOrSeries, Shape, final
from pandas.core.groupby import grouper
from pandas.core.indexes.api import Index
from pandas.core.series import Series
from pandas.util._decorators import cache_readonly

"""
Provide classes to perform the groupby aggregate operations.

These are not exposed to the user and provide implementations of the grouping
operations, primarily in cython. These classes (BaseGrouper and BinGrouper)
are contained *in* the SeriesGroupBy and DataFrameGroupBy objects.
"""

class WrappedCythonOp:
    """
    Dispatch logic for functions defined in _libs.groupby
    """

    cast_blocklist = ...
    def __init__(self, kind: str, how: str) -> None: ...
    _CYTHON_FUNCTIONS = ...
    _MASKED_CYTHON_FUNCTIONS = ...
    _cython_arity = ...
    def get_cython_func_and_vals(
        self, values: np.ndarray, is_numeric: bool
    ):  # -> tuple[Any, ndarray] | tuple[Any | None, ndarray]:
        """
        Find the appropriate cython function, casting if necessary.

        Parameters
        ----------
        values : np.ndarray
        is_numeric : bool

        Returns
        -------
        func : callable
        values : np.ndarray
        """
        ...
    def get_out_dtype(self, dtype: np.dtype) -> np.dtype: ...
    def uses_mask(self) -> bool: ...
    @final
    def cython_operation(
        self,
        *,
        values: ArrayLike,
        axis: int,
        min_count: int = ...,
        comp_ids: np.ndarray,
        ngroups: int,
        **kwargs
    ) -> ArrayLike:
        """
        Call our cython function, with appropriate pre- and post- processing.
        """
        ...

class BaseGrouper:
    """
    This is an internal Grouper class, which actually holds
    the generated groups

    Parameters
    ----------
    axis : Index
    groupings : Sequence[Grouping]
        all the grouping instances to handle in this grouper
        for example for grouper list to groupby, need to pass the list
    sort : bool, default True
        whether this grouper will give sorted result or not
    group_keys : bool, default True
    mutated : bool, default False
    indexer : np.ndarray[np.intp], optional
        the indexer created by Grouper
        some groupers (TimeGrouper) will sort its axis and its
        group_info is also sorted, so need the indexer to reorder

    """

    axis: Index
    def __init__(
        self,
        axis: Index,
        groupings: Sequence[grouper.Grouping],
        sort: bool = ...,
        group_keys: bool = ...,
        mutated: bool = ...,
        indexer: np.ndarray | None = ...,
        dropna: bool = ...,
    ) -> None: ...
    @property
    def groupings(self) -> list[grouper.Grouping]: ...
    @property
    def shape(self) -> Shape: ...
    def __iter__(self): ...
    @property
    def nkeys(self) -> int: ...
    def get_iterator(
        self, data: FrameOrSeries, axis: int = ...
    ) -> Iterator[tuple[Hashable, FrameOrSeries]]:
        """
        Groupby iterator

        Returns
        -------
        Generator yielding sequence of (name, subsetted object)
        for each group
        """
        ...
    @final
    def apply(self, f: F, data: FrameOrSeries, axis: int = ...): ...
    @cache_readonly
    def indices(
        self,
    ):  # -> () -> (() -> Unknown | dict[Hashable, ndarray]) | dict[str | tuple[Unknown, ...], ndarray]:
        """dict {group name -> group indices}"""
        ...
    @property
    def codes(self) -> list[np.ndarray]: ...
    @property
    def levels(self) -> list[Index]: ...
    @property
    def names(self) -> list[Hashable]: ...
    @final
    def size(self) -> Series:
        """
        Compute group sizes.
        """
        ...
    @cache_readonly
    def groups(self) -> dict[Hashable, np.ndarray]:
        """dict {group name -> group labels}"""
        ...
    @final
    @cache_readonly
    def is_monotonic(self) -> bool: ...
    @cache_readonly
    def group_info(self): ...
    @final
    @cache_readonly
    def codes_info(self) -> np.ndarray: ...
    @final
    @cache_readonly
    def ngroups(self) -> int: ...
    @property
    def reconstructed_codes(self) -> list[np.ndarray]: ...
    @cache_readonly
    def result_arraylike(self) -> ArrayLike:
        """
        Analogous to result_index, but returning an ndarray/ExtensionArray
        allowing us to retain ExtensionDtypes not supported by Index.
        """
        ...
    @cache_readonly
    def result_index(self) -> Index: ...
    @final
    def get_group_levels(self) -> list[ArrayLike]: ...
    @final
    def agg_series(self, obj: Series, func: F, preserve_dtype: bool = ...) -> ArrayLike:
        """
        Parameters
        ----------
        obj : Series
        func : function taking a Series and returning a scalar-like
        preserve_dtype : bool
            Whether the aggregation is known to be dtype-preserving.

        Returns
        -------
        np.ndarray or ExtensionArray
        """
        ...

class BinGrouper(BaseGrouper):
    """
    This is an internal Grouper class

    Parameters
    ----------
    bins : the split index of binlabels to group the item of axis
    binlabels : the label list
    mutated : bool, default False
    indexer : np.ndarray[np.intp]

    Examples
    --------
    bins: [2, 4, 6, 8, 10]
    binlabels: DatetimeIndex(['2005-01-01', '2005-01-03',
        '2005-01-05', '2005-01-07', '2005-01-09'],
        dtype='datetime64[ns]', freq='2D')

    the group_info, which contains the label of each item in grouped
    axis, the index of label in label list, group number, is

    (array([0, 0, 1, 1, 2, 2, 3, 3, 4, 4]), array([0, 1, 2, 3, 4]), 5)

    means that, the grouped axis has 10 items, can be grouped into 5
    labels, the first and second items belong to the first label, the
    third and forth items belong to the second label, and so on

    """

    bins: np.ndarray
    binlabels: Index
    mutated: bool
    def __init__(self, bins, binlabels, mutated: bool = ..., indexer=...) -> None: ...
    @cache_readonly
    def groups(self):  # -> dict[Any, Any]:
        """dict {group name -> group labels}"""
        ...
    @property
    def nkeys(self) -> int: ...
    def get_iterator(
        self, data: FrameOrSeries, axis: int = ...
    ):  # -> Generator[tuple[Any, Unknown] | tuple[ExtensionArray | Any | Index, Unknown], None, None]:
        """
        Groupby iterator

        Returns
        -------
        Generator yielding sequence of (name, subsetted object)
        for each group
        """
        ...
    @cache_readonly
    def indices(self): ...
    @cache_readonly
    def group_info(self): ...
    @cache_readonly
    def reconstructed_codes(self) -> list[np.ndarray]: ...
    @cache_readonly
    def result_index(self): ...
    @property
    def levels(self) -> list[Index]: ...
    @property
    def names(self) -> list[Hashable]: ...
    @property
    def groupings(self) -> list[grouper.Grouping]: ...

class DataSplitter(Generic[FrameOrSeries]):
    def __init__(
        self, data: FrameOrSeries, labels, ngroups: int, axis: int = ...
    ) -> None: ...
    @cache_readonly
    def slabels(self) -> np.ndarray: ...
    def __iter__(self): ...
    @cache_readonly
    def sorted_data(self) -> FrameOrSeries: ...

class SeriesSplitter(DataSplitter): ...

class FrameSplitter(DataSplitter):
    def fast_apply(self, f: F, sdata: FrameOrSeries, names): ...

def get_splitter(
    data: FrameOrSeries, labels: np.ndarray, ngroups: int, axis: int = ...
) -> DataSplitter: ...
