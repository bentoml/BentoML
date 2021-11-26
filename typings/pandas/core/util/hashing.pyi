from typing import TYPE_CHECKING, Hashable, Iterable, Iterator
import numpy as np
from pandas import Index, MultiIndex, Series
from pandas._typing import ArrayLike, FrameOrSeriesUnion

if TYPE_CHECKING: ...
_default_hash_key = ...

def combine_hash_arrays(arrays: Iterator[np.ndarray], num_items: int) -> np.ndarray: ...
def hash_pandas_object(
    obj: Index | FrameOrSeriesUnion,
    index: bool = ...,
    encoding: str = ...,
    hash_key: str | None = ...,
    categorize: bool = ...,
) -> Series: ...
def hash_tuples(
    vals: MultiIndex | Iterable[tuple[Hashable, ...]],
    encoding: str = ...,
    hash_key: str = ...,
) -> np.ndarray: ...
def hash_array(
    vals: ArrayLike, encoding: str = ..., hash_key: str = ..., categorize: bool = ...
) -> np.ndarray: ...
