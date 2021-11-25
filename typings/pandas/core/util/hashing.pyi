from typing import TYPE_CHECKING, Hashable, Iterable, Iterator

import numpy as np
from pandas import Index, MultiIndex, Series
from pandas._typing import ArrayLike, FrameOrSeriesUnion

"""
data hash pandas / numpy objects
"""
if TYPE_CHECKING: ...
_default_hash_key = ...

def combine_hash_arrays(arrays: Iterator[np.ndarray], num_items: int) -> np.ndarray:
    """
    Parameters
    ----------
    arrays : Iterator[np.ndarray]
    num_items : int

    Returns
    -------
    np.ndarray[uint64]

    Should be the same as CPython's tupleobject.c
    """
    ...

def hash_pandas_object(
    obj: Index | FrameOrSeriesUnion,
    index: bool = ...,
    encoding: str = ...,
    hash_key: str | None = ...,
    categorize: bool = ...,
) -> Series:
    """
    Return a data hash of the Index/Series/DataFrame.

    Parameters
    ----------
    obj : Index, Series, or DataFrame
    index : bool, default True
        Include the index in the hash (if Series/DataFrame).
    encoding : str, default 'utf8'
        Encoding for data & key when strings.
    hash_key : str, default _default_hash_key
        Hash_key for string key to encode.
    categorize : bool, default True
        Whether to first categorize object arrays before hashing. This is more
        efficient when the array contains duplicate values.

    Returns
    -------
    Series of uint64, same length as the object
    """
    ...

def hash_tuples(
    vals: MultiIndex | Iterable[tuple[Hashable, ...]],
    encoding: str = ...,
    hash_key: str = ...,
) -> np.ndarray:
    """
    Hash an MultiIndex / listlike-of-tuples efficiently.

    Parameters
    ----------
    vals : MultiIndex or listlike-of-tuples
    encoding : str, default 'utf8'
    hash_key : str, default _default_hash_key

    Returns
    -------
    ndarray[np.uint64] of hashed values
    """
    ...

def hash_array(
    vals: ArrayLike, encoding: str = ..., hash_key: str = ..., categorize: bool = ...
) -> np.ndarray:
    """
    Given a 1d array, return an array of deterministic integers.

    Parameters
    ----------
    vals : ndarray or ExtensionArray
    encoding : str, default 'utf8'
        Encoding for data & key when strings.
    hash_key : str, default _default_hash_key
        Hash_key for string key to encode.
    categorize : bool, default True
        Whether to first categorize object arrays before hashing. This is more
        efficient when the array contains duplicate values.

    Returns
    -------
    ndarray[np.uint64, ndim=1]
        Hashed values, same length as the vals.
    """
    ...
