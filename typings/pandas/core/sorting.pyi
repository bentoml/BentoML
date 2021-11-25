from typing import TYPE_CHECKING, Callable, Iterable, Sequence

import numpy as np
from pandas._typing import IndexKeyFunc, Shape
from pandas.core.indexes.base import Index

""" miscellaneous sorting / groupby utilities """
if TYPE_CHECKING: ...

def get_indexer_indexer(
    target: Index,
    level: str | int | list[str] | list[int],
    ascending: Sequence[bool | int] | bool | int,
    kind: str,
    na_position: str,
    sort_remaining: bool,
    key: IndexKeyFunc,
) -> np.ndarray | None:
    """
    Helper method that return the indexer according to input parameters for
    the sort_index method of DataFrame and Series.

    Parameters
    ----------
    target : Index
    level : int or level name or list of ints or list of level names
    ascending : bool or list of bools, default True
    kind : {'quicksort', 'mergesort', 'heapsort', 'stable'}, default 'quicksort'
    na_position : {'first', 'last'}, default 'last'
    sort_remaining : bool, default True
    key : callable, optional

    Returns
    -------
    Optional[ndarray]
        The indexer for the new index.
    """
    ...

def get_group_index(labels, shape: Shape, sort: bool, xnull: bool):  # -> Any:
    """
    For the particular label_list, gets the offsets into the hypothetical list
    representing the totally ordered cartesian product of all possible label
    combinations, *as long as* this space fits within int64 bounds;
    otherwise, though group indices identify unique combinations of
    labels, they cannot be deconstructed.
    - If `sort`, rank of returned ids preserve lexical ranks of labels.
      i.e. returned id's can be used to do lexical sort on labels;
    - If `xnull` nulls (-1 labels) are passed through.

    Parameters
    ----------
    labels : sequence of arrays
        Integers identifying levels at each location
    shape : tuple[int, ...]
        Number of unique levels at each location
    sort : bool
        If the ranks of returned ids should match lexical ranks of labels
    xnull : bool
        If true nulls are excluded. i.e. -1 values in the labels are
        passed through.

    Returns
    -------
    An array of type int64 where two elements are equal if their corresponding
    labels are equal at all location.

    Notes
    -----
    The length of `labels` and `shape` must be identical.
    """
    ...

def get_compressed_ids(labels, sizes: Shape) -> tuple[np.ndarray, np.ndarray]:
    """
    Group_index is offsets into cartesian product of all possible labels. This
    space can be huge, so this function compresses it, by computing offsets
    (comp_ids) into the list of unique labels (obs_group_ids).

    Parameters
    ----------
    labels : list of label arrays
    sizes : tuple[int] of size of the levels

    Returns
    -------
    np.ndarray[np.intp]
        comp_ids
    np.ndarray[np.int64]
        obs_group_ids
    """
    ...

def is_int64_overflow_possible(shape) -> bool: ...
def decons_group_index(comp_labels, shape): ...
def decons_obs_group_ids(
    comp_ids: np.ndarray, obs_ids, shape, labels, xnull: bool
):  # -> list[Unknown]:
    """
    Reconstruct labels from observed group ids.

    Parameters
    ----------
    comp_ids : np.ndarray[np.intp]
    xnull : bool
        If nulls are excluded; i.e. -1 labels are passed through.
    """
    ...

def indexer_from_factorized(labels, shape: Shape, compress: bool = ...) -> np.ndarray: ...
def lexsort_indexer(
    keys, orders=..., na_position: str = ..., key: Callable | None = ...
) -> np.ndarray:
    """
    Performs lexical sorting on a set of keys

    Parameters
    ----------
    keys : sequence of arrays
        Sequence of ndarrays to be sorted by the indexer
    orders : bool or list of booleans, optional
        Determines the sorting order for each element in keys. If a list,
        it must be the same length as keys. This determines whether the
        corresponding element in keys should be sorted in ascending
        (True) or descending (False) order. if bool, applied to all
        elements as above. if None, defaults to True.
    na_position : {'first', 'last'}, default 'last'
        Determines placement of NA elements in the sorted list ("last" or "first")
    key : Callable, optional
        Callable key function applied to every element in keys before sorting

        .. versionadded:: 1.0.0

    Returns
    -------
    np.ndarray[np.intp]
    """
    ...

def nargsort(
    items,
    kind: str = ...,
    ascending: bool = ...,
    na_position: str = ...,
    key: Callable | None = ...,
    mask: np.ndarray | None = ...,
):  # -> ndarray | Any:
    """
    Intended to be a drop-in replacement for np.argsort which handles NaNs.

    Adds ascending, na_position, and key parameters.

    (GH #6399, #5231, #27237)

    Parameters
    ----------
    kind : str, default 'quicksort'
    ascending : bool, default True
    na_position : {'first', 'last'}, default 'last'
    key : Optional[Callable], default None
    mask : Optional[np.ndarray], default None
        Passed when called by ExtensionArray.argsort.

    Returns
    -------
    np.ndarray[np.intp]
    """
    ...

def nargminmax(values, method: str, axis: int = ...):  # -> ndarray | Any | int:
    """
    Implementation of np.argmin/argmax but for ExtensionArray and which
    handles missing values.

    Parameters
    ----------
    values : ExtensionArray
    method : {"argmax", "argmin"}
    axis : int, default 0

    Returns
    -------
    int
    """
    ...

def ensure_key_mapped(
    values, key: Callable | None, levels=...
):  # -> MultiIndex | Index | Any:
    """
    Applies a callable key function to the values function and checks
    that the resulting value has the same shape. Can be called on Index
    subclasses, Series, DataFrames, or ndarrays.

    Parameters
    ----------
    values : Series, DataFrame, Index subclass, or ndarray
    key : Optional[Callable], key to be called on the values array
    levels : Optional[List], if values is a MultiIndex, list of levels to
    apply the key to.
    """
    ...

def get_flattened_list(
    comp_ids: np.ndarray,
    ngroups: int,
    levels: Iterable[Index],
    labels: Iterable[np.ndarray],
) -> list[tuple]:
    """Map compressed group id -> key tuple."""
    ...

def get_indexer_dict(
    label_list: list[np.ndarray], keys: list[Index]
) -> dict[str | tuple, np.ndarray]:
    """
    Returns
    -------
    dict:
        Labels mapped to indexers.
    """
    ...

def get_group_index_sorter(
    group_index: np.ndarray, ngroups: int | None = ...
) -> np.ndarray:
    """
    algos.groupsort_indexer implements `counting sort` and it is at least
    O(ngroups), where
        ngroups = prod(shape)
        shape = map(len, keys)
    that is, linear in the number of combinations (cartesian product) of unique
    values of groupby keys. This can be huge when doing multi-key groupby.
    np.argsort(kind='mergesort') is O(count x log(count)) where count is the
    length of the data-frame;
    Both algorithms are `stable` sort and that is necessary for correctness of
    groupby operations. e.g. consider:
        df.groupby(key)[col].transform('first')

    Parameters
    ----------
    group_index : np.ndarray[np.intp]
        signed integer dtype
    ngroups : int or None, default None

    Returns
    -------
    np.ndarray[np.intp]
    """
    ...

def compress_group_index(
    group_index: np.ndarray, sort: bool = ...
) -> tuple[np.ndarray, np.ndarray]:
    """
    Group_index is offsets into cartesian product of all possible labels. This
    space can be huge, so this function compresses it, by computing offsets
    (comp_ids) into the list of unique labels (obs_group_ids).
    """
    ...
