from typing import Any, Generic, List, Literal, SupportsIndex, Tuple, TypeVar, overload
from numpy import _CastingKind, _OrderKACF, ufunc
from numpy.typing import NDArray
from ._array_like import ArrayLike, _ArrayLikeBool_co, _ArrayLikeInt_co
from ._dtype_like import DTypeLike
from ._scalars import _ScalarLike_co
from ._shape import _ShapeLike

_T = TypeVar("_T")
_2Tuple = Tuple[_T, _T]
_3Tuple = Tuple[_T, _T, _T]
_4Tuple = Tuple[_T, _T, _T, _T]
_NTypes = TypeVar("_NTypes", bound=int)
_IDType = TypeVar("_IDType", bound=Any)
_NameType = TypeVar("_NameType", bound=str)

class _UFunc_Nin1_Nout1(ufunc, Generic[_NameType, _NTypes, _IDType]):
    @property
    def __name__(self) -> _NameType: ...
    @property
    def ntypes(self) -> _NTypes: ...
    @property
    def identity(self) -> _IDType: ...
    @property
    def nin(self) -> Literal[1]: ...
    @property
    def nout(self) -> Literal[1]: ...
    @property
    def nargs(self) -> Literal[2]: ...
    @property
    def signature(self) -> None: ...
    @property
    def reduce(self) -> None: ...
    @property
    def accumulate(self) -> None: ...
    @property
    def reduceat(self) -> None: ...
    @property
    def outer(self) -> None: ...
    @overload
    def __call__(
        self,
        __x1: _ScalarLike_co,
        out: None = ...,
        *,
        where: None | _ArrayLikeBool_co = ...,
        casting: _CastingKind = ...,
        order: _OrderKACF = ...,
        dtype: DTypeLike = ...,
        subok: bool = ...,
        signature: str | _2Tuple[None | str] = ...,
        extobj: List[Any] = ...,
    ) -> Any: ...
    @overload
    def __call__(
        self,
        __x1: ArrayLike,
        out: None | NDArray[Any] | Tuple[NDArray[Any]] = ...,
        *,
        where: None | _ArrayLikeBool_co = ...,
        casting: _CastingKind = ...,
        order: _OrderKACF = ...,
        dtype: DTypeLike = ...,
        subok: bool = ...,
        signature: str | _2Tuple[None | str] = ...,
        extobj: List[Any] = ...,
    ) -> NDArray[Any]: ...
    def at(self, a: NDArray[Any], indices: _ArrayLikeInt_co, /) -> None: ...

class _UFunc_Nin2_Nout1(ufunc, Generic[_NameType, _NTypes, _IDType]):
    @property
    def __name__(self) -> _NameType: ...
    @property
    def ntypes(self) -> _NTypes: ...
    @property
    def identity(self) -> _IDType: ...
    @property
    def nin(self) -> Literal[2]: ...
    @property
    def nout(self) -> Literal[1]: ...
    @property
    def nargs(self) -> Literal[3]: ...
    @property
    def signature(self) -> None: ...
    @overload
    def __call__(
        self,
        __x1: _ScalarLike_co,
        __x2: _ScalarLike_co,
        out: None = ...,
        *,
        where: None | _ArrayLikeBool_co = ...,
        casting: _CastingKind = ...,
        order: _OrderKACF = ...,
        dtype: DTypeLike = ...,
        subok: bool = ...,
        signature: str | _3Tuple[None | str] = ...,
        extobj: List[Any] = ...,
    ) -> Any: ...
    @overload
    def __call__(
        self,
        __x1: ArrayLike,
        __x2: ArrayLike,
        out: None | NDArray[Any] | Tuple[NDArray[Any]] = ...,
        *,
        where: None | _ArrayLikeBool_co = ...,
        casting: _CastingKind = ...,
        order: _OrderKACF = ...,
        dtype: DTypeLike = ...,
        subok: bool = ...,
        signature: str | _3Tuple[None | str] = ...,
        extobj: List[Any] = ...,
    ) -> NDArray[Any]: ...
    def at(
        self, a: NDArray[Any], indices: _ArrayLikeInt_co, b: ArrayLike, /
    ) -> None: ...
    def reduce(
        self,
        array: ArrayLike,
        axis: None | _ShapeLike = ...,
        dtype: DTypeLike = ...,
        out: None | NDArray[Any] = ...,
        keepdims: bool = ...,
        initial: Any = ...,
        where: _ArrayLikeBool_co = ...,
    ) -> Any: ...
    def accumulate(
        self,
        array: ArrayLike,
        axis: SupportsIndex = ...,
        dtype: DTypeLike = ...,
        out: None | NDArray[Any] = ...,
    ) -> NDArray[Any]: ...
    def reduceat(
        self,
        array: ArrayLike,
        indices: _ArrayLikeInt_co,
        axis: SupportsIndex = ...,
        dtype: DTypeLike = ...,
        out: None | NDArray[Any] = ...,
    ) -> NDArray[Any]: ...
    @overload
    def outer(
        self,
        A: _ScalarLike_co,
        B: _ScalarLike_co,
        /,
        *,
        out: None = ...,
        where: None | _ArrayLikeBool_co = ...,
        casting: _CastingKind = ...,
        order: _OrderKACF = ...,
        dtype: DTypeLike = ...,
        subok: bool = ...,
        signature: str | _3Tuple[None | str] = ...,
        extobj: List[Any] = ...,
    ) -> Any: ...
    @overload
    def outer(
        self,
        A: ArrayLike,
        B: ArrayLike,
        /,
        *,
        out: None | NDArray[Any] | Tuple[NDArray[Any]] = ...,
        where: None | _ArrayLikeBool_co = ...,
        casting: _CastingKind = ...,
        order: _OrderKACF = ...,
        dtype: DTypeLike = ...,
        subok: bool = ...,
        signature: str | _3Tuple[None | str] = ...,
        extobj: List[Any] = ...,
    ) -> NDArray[Any]: ...

class _UFunc_Nin1_Nout2(ufunc, Generic[_NameType, _NTypes, _IDType]):
    @property
    def __name__(self) -> _NameType: ...
    @property
    def ntypes(self) -> _NTypes: ...
    @property
    def identity(self) -> _IDType: ...
    @property
    def nin(self) -> Literal[1]: ...
    @property
    def nout(self) -> Literal[2]: ...
    @property
    def nargs(self) -> Literal[3]: ...
    @property
    def signature(self) -> None: ...
    @property
    def at(self) -> None: ...
    @property
    def reduce(self) -> None: ...
    @property
    def accumulate(self) -> None: ...
    @property
    def reduceat(self) -> None: ...
    @property
    def outer(self) -> None: ...
    @overload
    def __call__(
        self,
        __x1: _ScalarLike_co,
        __out1: None = ...,
        __out2: None = ...,
        *,
        where: None | _ArrayLikeBool_co = ...,
        casting: _CastingKind = ...,
        order: _OrderKACF = ...,
        dtype: DTypeLike = ...,
        subok: bool = ...,
        signature: str | _3Tuple[None | str] = ...,
        extobj: List[Any] = ...,
    ) -> _2Tuple[Any]: ...
    @overload
    def __call__(
        self,
        __x1: ArrayLike,
        __out1: None | NDArray[Any] = ...,
        __out2: None | NDArray[Any] = ...,
        *,
        out: _2Tuple[NDArray[Any]] = ...,
        where: None | _ArrayLikeBool_co = ...,
        casting: _CastingKind = ...,
        order: _OrderKACF = ...,
        dtype: DTypeLike = ...,
        subok: bool = ...,
        signature: str | _3Tuple[None | str] = ...,
        extobj: List[Any] = ...,
    ) -> _2Tuple[NDArray[Any]]: ...

class _UFunc_Nin2_Nout2(ufunc, Generic[_NameType, _NTypes, _IDType]):
    @property
    def __name__(self) -> _NameType: ...
    @property
    def ntypes(self) -> _NTypes: ...
    @property
    def identity(self) -> _IDType: ...
    @property
    def nin(self) -> Literal[2]: ...
    @property
    def nout(self) -> Literal[2]: ...
    @property
    def nargs(self) -> Literal[4]: ...
    @property
    def signature(self) -> None: ...
    @property
    def at(self) -> None: ...
    @property
    def reduce(self) -> None: ...
    @property
    def accumulate(self) -> None: ...
    @property
    def reduceat(self) -> None: ...
    @property
    def outer(self) -> None: ...
    @overload
    def __call__(
        self,
        __x1: _ScalarLike_co,
        __x2: _ScalarLike_co,
        __out1: None = ...,
        __out2: None = ...,
        *,
        where: None | _ArrayLikeBool_co = ...,
        casting: _CastingKind = ...,
        order: _OrderKACF = ...,
        dtype: DTypeLike = ...,
        subok: bool = ...,
        signature: str | _4Tuple[None | str] = ...,
        extobj: List[Any] = ...,
    ) -> _2Tuple[Any]: ...
    @overload
    def __call__(
        self,
        __x1: ArrayLike,
        __x2: ArrayLike,
        __out1: None | NDArray[Any] = ...,
        __out2: None | NDArray[Any] = ...,
        *,
        out: _2Tuple[NDArray[Any]] = ...,
        where: None | _ArrayLikeBool_co = ...,
        casting: _CastingKind = ...,
        order: _OrderKACF = ...,
        dtype: DTypeLike = ...,
        subok: bool = ...,
        signature: str | _4Tuple[None | str] = ...,
        extobj: List[Any] = ...,
    ) -> _2Tuple[NDArray[Any]]: ...

class _GUFunc_Nin2_Nout1(ufunc, Generic[_NameType, _NTypes, _IDType]):
    @property
    def __name__(self) -> _NameType: ...
    @property
    def ntypes(self) -> _NTypes: ...
    @property
    def identity(self) -> _IDType: ...
    @property
    def nin(self) -> Literal[2]: ...
    @property
    def nout(self) -> Literal[1]: ...
    @property
    def nargs(self) -> Literal[3]: ...
    @property
    def signature(self) -> Literal["(n?,k),(k,m?)->(n?,m?)"]: ...
    @property
    def reduce(self) -> None: ...
    @property
    def accumulate(self) -> None: ...
    @property
    def reduceat(self) -> None: ...
    @property
    def outer(self) -> None: ...
    @property
    def at(self) -> None: ...
    @overload
    def __call__(
        self,
        __x1: ArrayLike,
        __x2: ArrayLike,
        out: None = ...,
        *,
        casting: _CastingKind = ...,
        order: _OrderKACF = ...,
        dtype: DTypeLike = ...,
        subok: bool = ...,
        signature: str | _3Tuple[None | str] = ...,
        extobj: List[Any] = ...,
        axes: List[_2Tuple[SupportsIndex]] = ...,
    ) -> Any: ...
    @overload
    def __call__(
        self,
        __x1: ArrayLike,
        __x2: ArrayLike,
        out: NDArray[Any] | Tuple[NDArray[Any]],
        *,
        casting: _CastingKind = ...,
        order: _OrderKACF = ...,
        dtype: DTypeLike = ...,
        subok: bool = ...,
        signature: str | _3Tuple[None | str] = ...,
        extobj: List[Any] = ...,
        axes: List[_2Tuple[SupportsIndex]] = ...,
    ) -> NDArray[Any]: ...
