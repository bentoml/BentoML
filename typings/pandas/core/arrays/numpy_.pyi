from typing import Any

import numpy as np
from numpy.lib.mixins import NDArrayOperatorsMixin
from pandas._typing import NpDtype, Scalar
from pandas.core.arraylike import OpsMixin
from pandas.core.arrays._mixins import NDArrayBackedExtensionArray
from pandas.core.dtypes.dtypes import PandasDtype
from pandas.core.strings.object_array import ObjectStringArrayMixin

class PandasArray(
    OpsMixin, NDArrayBackedExtensionArray, NDArrayOperatorsMixin, ObjectStringArrayMixin
):
    _typ = ...
    __array_priority__ = ...
    _ndarray: np.ndarray
    _dtype: PandasDtype
    def __init__(self, values: np.ndarray | PandasArray, copy: bool = ...) -> None: ...
    @property
    def dtype(self) -> PandasDtype: ...
    def __array__(self, dtype: NpDtype | None = ...) -> np.ndarray: ...
    _HANDLED_TYPES = ...
    def __array_ufunc__(self, ufunc: np.ufunc, method: str, *inputs, **kwargs): ...
    def isna(self) -> np.ndarray: ...
    def any(
        self,
        *,
        axis: int | None = ...,
        out=...,
        keepdims: bool = ...,
        skipna: bool = ...
    ): ...
    def all(
        self,
        *,
        axis: int | None = ...,
        out=...,
        keepdims: bool = ...,
        skipna: bool = ...
    ): ...
    def min(
        self, *, axis: int | None = ..., skipna: bool = ..., **kwargs
    ) -> Scalar: ...
    def max(
        self, *, axis: int | None = ..., skipna: bool = ..., **kwargs
    ) -> Scalar: ...
    def sum(
        self, *, axis: int | None = ..., skipna: bool = ..., min_count=..., **kwargs
    ) -> Scalar: ...
    def prod(
        self, *, axis: int | None = ..., skipna: bool = ..., min_count=..., **kwargs
    ) -> Scalar: ...
    def mean(
        self,
        *,
        axis: int | None = ...,
        dtype: NpDtype | None = ...,
        out=...,
        keepdims: bool = ...,
        skipna: bool = ...
    ): ...
    def median(
        self,
        *,
        axis: int | None = ...,
        out=...,
        overwrite_input: bool = ...,
        keepdims: bool = ...,
        skipna: bool = ...
    ): ...
    def std(
        self,
        *,
        axis: int | None = ...,
        dtype: NpDtype | None = ...,
        out=...,
        ddof=...,
        keepdims: bool = ...,
        skipna: bool = ...
    ): ...
    def var(
        self,
        *,
        axis: int | None = ...,
        dtype: NpDtype | None = ...,
        out=...,
        ddof=...,
        keepdims: bool = ...,
        skipna: bool = ...
    ): ...
    def sem(
        self,
        *,
        axis: int | None = ...,
        dtype: NpDtype | None = ...,
        out=...,
        ddof=...,
        keepdims: bool = ...,
        skipna: bool = ...
    ): ...
    def kurt(
        self,
        *,
        axis: int | None = ...,
        dtype: NpDtype | None = ...,
        out=...,
        keepdims: bool = ...,
        skipna: bool = ...
    ): ...
    def skew(
        self,
        *,
        axis: int | None = ...,
        dtype: NpDtype | None = ...,
        out=...,
        keepdims: bool = ...,
        skipna: bool = ...
    ): ...
    def to_numpy(
        self, dtype: NpDtype | None = ..., copy: bool = ..., na_value: Scalar=...
    ) -> np.ndarray[Any, np.dtype[Any]]: ...
    def __invert__(self) -> PandasArray: ...
    _arith_method = ...
    _str_na_value = ...
