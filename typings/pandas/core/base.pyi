import textwrap
from typing import TYPE_CHECKING, Any, Generic, Hashable
import numpy as np
from pandas._typing import Dtype, DtypeObj, FrameOrSeries, IndexLabel, Shape, final
from pandas.core import algorithms
from pandas.core.accessor import DirNamesMixin
from pandas.core.arraylike import OpsMixin
from pandas.core.arrays import ExtensionArray
from pandas.util._decorators import cache_readonly, doc

if TYPE_CHECKING: ...
_shared_docs: dict[str, str] = ...
_indexops_doc_kwargs = ...
_T = ...

class PandasObject(DirNamesMixin):
    _cache: dict[str, Any]
    def __repr__(self) -> str: ...
    def __sizeof__(self) -> int: ...

class NoNewAttributesMixin:
    def __setattr__(self, key: str, value): ...

class DataError(Exception): ...
class SpecificationError(Exception): ...

class SelectionMixin(Generic[FrameOrSeries]):
    obj: FrameOrSeries
    _selection: IndexLabel | None = ...
    exclusions: frozenset[Hashable]
    _internal_names = ...
    _internal_names_set = ...
    @final
    @cache_readonly
    def ndim(self) -> int: ...
    def __getitem__(self, key): ...
    def aggregate(self, func, *args, **kwargs): ...
    agg = ...

class IndexOpsMixin(OpsMixin):
    __array_priority__ = ...
    _hidden_attrs: frozenset[str] = ...
    @property
    def dtype(self) -> DtypeObj: ...
    def transpose(self: _T, *args, **kwargs) -> _T: ...
    T = ...
    @property
    def shape(self) -> Shape: ...
    def __len__(self) -> int: ...
    @property
    def ndim(self) -> int: ...
    def item(self): ...
    @property
    def nbytes(self) -> int: ...
    @property
    def size(self) -> int: ...
    @property
    def array(self) -> ExtensionArray: ...
    def to_numpy(
        self, dtype: Dtype | None = ..., copy: bool = ..., na_value=..., **kwargs
    ) -> np.ndarray[Any, np.dtype[Any]]: ...
    @property
    def empty(self) -> bool: ...
    def max(self, axis=..., skipna: bool = ..., *args, **kwargs): ...
    @doc(op="max", oppose="min", value="largest")
    def argmax(self, axis=..., skipna: bool = ..., *args, **kwargs) -> int: ...
    def min(self, axis=..., skipna: bool = ..., *args, **kwargs): ...
    @doc(argmax, op="min", oppose="max", value="smallest")
    def argmin(self, axis=..., skipna=..., *args, **kwargs) -> int: ...
    def tolist(self): ...
    to_list = ...
    def __iter__(self): ...
    @cache_readonly
    def hasnans(self) -> bool: ...
    def isna(self): ...
    def value_counts(
        self,
        normalize: bool = ...,
        sort: bool = ...,
        ascending: bool = ...,
        bins=...,
        dropna: bool = ...,
    ): ...
    def unique(self): ...
    def nunique(self, dropna: bool = ...) -> int: ...
    @property
    def is_unique(self) -> bool: ...
    @property
    def is_monotonic(self) -> bool: ...
    @property
    def is_monotonic_increasing(self) -> bool: ...
    @property
    def is_monotonic_decreasing(self) -> bool: ...
    @doc(
        algorithms.factorize,
        values="",
        order="",
        size_hint="",
        sort=textwrap.dedent(
            """            sort : bool, default False
                Sort `uniques` and shuffle `codes` to maintain the
                relationship.
            """
        ),
    )
    def factorize(self, sort: bool = ..., na_sentinel: int | None = ...): ...
    @doc(_shared_docs["searchsorted"], klass="Index")
    def searchsorted(self, value, side=..., sorter=...) -> np.ndarray: ...
