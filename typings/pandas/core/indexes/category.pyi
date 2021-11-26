from typing import Any, Hashable
import numpy as np
from pandas._typing import Dtype
from pandas.core import accessor
from pandas.core.arrays.categorical import Categorical
from pandas.core.indexes.base import Index, _index_shared_docs
from pandas.core.indexes.extension import NDArrayBackedExtensionIndex, inherit_names
from pandas.util._decorators import Appender, doc

_index_doc_kwargs: dict[str, str] = ...

@inherit_names(
    [
        "argsort",
        "_internal_get_values",
        "tolist",
        "codes",
        "categories",
        "ordered",
        "_reverse_indexer",
        "searchsorted",
        "is_dtype_equal",
        "min",
        "max",
    ],
    Categorical,
)
@accessor.delegate_names(
    delegate=Categorical,
    accessors=[
        "rename_categories",
        "reorder_categories",
        "add_categories",
        "remove_categories",
        "remove_unused_categories",
        "set_categories",
        "as_ordered",
        "as_unordered",
    ],
    typ="method",
    overwrite=True,
)
class CategoricalIndex(NDArrayBackedExtensionIndex, accessor.PandasDelegate):
    _typ = ...
    _data_cls = Categorical
    codes: np.ndarray
    categories: Index
    _data: Categorical
    _values: Categorical
    _attributes = ...
    def __new__(
        cls,
        data=...,
        categories=...,
        ordered=...,
        dtype: Dtype | None = ...,
        copy: bool = ...,
        name: Hashable = ...,
    ) -> CategoricalIndex: ...
    def equals(self, other: object) -> bool: ...
    @property
    def inferred_type(self) -> str: ...
    @doc(Index.__contains__)
    def __contains__(self, key: Any) -> bool: ...
    @doc(Index.fillna)
    def fillna(self, value, downcast=...): ...
    def reindex(
        self, target, method=..., level=..., limit=..., tolerance=...
    ) -> tuple[Index, np.ndarray | None]: ...
    @Appender(_index_shared_docs["get_indexer_non_unique"] % _index_doc_kwargs)
    def get_indexer_non_unique(self, target) -> tuple[np.ndarray, np.ndarray]: ...
    def take_nd(self, *args, **kwargs): ...
    def map(self, mapper): ...
