from typing import TYPE_CHECKING, Any, Sequence
import numpy as np
from pandas import Series
from pandas._typing import NpDtype, PositionalIndexer
from pandas.compat import pa_version_under1p0
from pandas.core.arraylike import OpsMixin
from pandas.core.arrays.base import ExtensionArray
from pandas.core.arrays.string_ import BaseStringArray, StringDtype
from pandas.core.strings.object_array import ObjectStringArrayMixin
from pandas.util._decorators import doc

if notpa_version_under1p0:
    ARROW_CMP_FUNCS = ...
if TYPE_CHECKING: ...

class ArrowStringArray(OpsMixin, BaseStringArray, ObjectStringArrayMixin):
    def __init__(self, values) -> None: ...
    @property
    def dtype(self) -> StringDtype: ...
    def __array__(self, dtype: NpDtype | None = ...) -> np.ndarray: ...
    def __arrow_array__(self, type=...): ...
    def to_numpy(
        self, dtype: NpDtype | None = ..., copy: bool = ..., na_value: Any = ...
    ) -> np.ndarray[Any, np.dtype[Any]]: ...
    def __len__(self) -> int: ...
    @doc(ExtensionArray.factorize)
    def factorize(
        self, na_sentinel: int = ...
    ) -> tuple[np.ndarray, ExtensionArray]: ...
    def __getitem__(self, item: PositionalIndexer) -> Any: ...
    def fillna(self, value=..., method=..., limit=...): ...
    @property
    def nbytes(self) -> int: ...
    def isna(self) -> np.ndarray: ...
    def copy(self) -> ArrowStringArray: ...
    def __setitem__(self, key: int | slice | np.ndarray, value: Any) -> None: ...
    def take(
        self, indices: Sequence[int], allow_fill: bool = ..., fill_value: Any = ...
    ): ...
    def isin(self, values): ...
    def value_counts(self, dropna: bool = ...) -> Series: ...
    def astype(self, dtype, copy=...): ...
    _str_na_value = ...
