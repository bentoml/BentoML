import datetime as dt
import os
from typing import Any, Callable, Final, Iterable, List
from typing import Literal as L
from typing import (
    Optional,
    Protocol,
    Sequence,
    SupportsIndex,
    Tuple,
    Type,
    TypeVar,
    Union,
    final,
    overload,
)
from numpy import (
    _CastingKind,
    _CopyMode,
    _IOProtocol,
    _ModeKind,
    _NDIterFlagsKind,
    _NDIterOpFlagsKind,
    _OrderCF,
    _OrderKACF,
    _SupportsBuffer,
    bool_,
)
from numpy import broadcast as broadcast
from numpy import busdaycalendar as busdaycalendar
from numpy import complexfloating, datetime64
from numpy import dtype as dtype
from numpy import float64, floating, generic, int_, intp
from numpy import ndarray as ndarray
from numpy import nditer as nditer
from numpy import signedinteger, str_, timedelta64, ufunc, uint8, unsignedinteger
from numpy.typing import (
    ArrayLike,
    DTypeLike,
    NDArray,
    _ArrayLikeBool_co,
    _ArrayLikeBytes_co,
    _ArrayLikeComplex_co,
    _ArrayLikeDT64_co,
    _ArrayLikeFloat_co,
    _ArrayLikeInt_co,
    _ArrayLikeObject_co,
    _ArrayLikeStr_co,
    _ArrayLikeTD64_co,
    _ArrayLikeUInt_co,
    _FiniteNestedSequence,
    _FloatLike_co,
    _IntLike_co,
    _ScalarLike_co,
    _ShapeLike,
    _SupportsArray,
    _SupportsDType,
    _TD64Like_co,
)

_SCT = TypeVar("_SCT", bound=generic)
_ArrayType = TypeVar("_ArrayType", bound=NDArray[Any])
_DTypeLike = Union[dtype[_SCT], Type[_SCT], _SupportsDType[dtype[_SCT]]]
_ArrayLike = _FiniteNestedSequence[_SupportsArray[dtype[_SCT]]]
_UnitKind = L["Y", "M", "D", "h", "m", "s", "ms", "us", "μs", "ns", "ps", "fs", "as"]
_RollKind = L[
    "nat",
    "forward",
    "following",
    "backward",
    "preceding",
    "modifiedfollowing",
    "modifiedpreceding",
]
__all__: List[str]
ALLOW_THREADS: Final[int]
BUFSIZE: L[8192]
CLIP: L[0]
WRAP: L[1]
RAISE: L[2]
MAXDIMS: L[32]
MAY_SHARE_BOUNDS: L[0]
MAY_SHARE_EXACT: L[-1]
tracemalloc_domain: L[389047]

@overload
def empty_like(
    prototype: _ArrayType,
    dtype: None = ...,
    order: _OrderKACF = ...,
    subok: bool = ...,
    shape: Optional[_ShapeLike] = ...,
) -> _ArrayType: ...
@overload
def empty_like(
    prototype: _ArrayLike[_SCT],
    dtype: None = ...,
    order: _OrderKACF = ...,
    subok: bool = ...,
    shape: Optional[_ShapeLike] = ...,
) -> NDArray[_SCT]: ...
@overload
def empty_like(
    prototype: object,
    dtype: None = ...,
    order: _OrderKACF = ...,
    subok: bool = ...,
    shape: Optional[_ShapeLike] = ...,
) -> NDArray[Any]: ...
@overload
def empty_like(
    prototype: Any,
    dtype: _DTypeLike[_SCT],
    order: _OrderKACF = ...,
    subok: bool = ...,
    shape: Optional[_ShapeLike] = ...,
) -> NDArray[_SCT]: ...
@overload
def empty_like(
    prototype: Any,
    dtype: DTypeLike,
    order: _OrderKACF = ...,
    subok: bool = ...,
    shape: Optional[_ShapeLike] = ...,
) -> NDArray[Any]: ...
@overload
def array(
    object: _ArrayType,
    dtype: None = ...,
    *,
    copy: bool | _CopyMode = ...,
    order: _OrderKACF = ...,
    subok: L[True],
    ndmin: int = ...,
    like: ArrayLike = ...,
) -> _ArrayType: ...
@overload
def array(
    object: _ArrayLike[_SCT],
    dtype: None = ...,
    *,
    copy: bool | _CopyMode = ...,
    order: _OrderKACF = ...,
    subok: bool = ...,
    ndmin: int = ...,
    like: ArrayLike = ...,
) -> NDArray[_SCT]: ...
@overload
def array(
    object: object,
    dtype: None = ...,
    *,
    copy: bool | _CopyMode = ...,
    order: _OrderKACF = ...,
    subok: bool = ...,
    ndmin: int = ...,
    like: ArrayLike = ...,
) -> NDArray[Any]: ...
@overload
def array(
    object: Any,
    dtype: _DTypeLike[_SCT],
    *,
    copy: bool | _CopyMode = ...,
    order: _OrderKACF = ...,
    subok: bool = ...,
    ndmin: int = ...,
    like: ArrayLike = ...,
) -> NDArray[_SCT]: ...
@overload
def array(
    object: Any,
    dtype: DTypeLike,
    *,
    copy: bool | _CopyMode = ...,
    order: _OrderKACF = ...,
    subok: bool = ...,
    ndmin: int = ...,
    like: ArrayLike = ...,
) -> NDArray[Any]: ...
@overload
def zeros(
    shape: _ShapeLike,
    dtype: None = ...,
    order: _OrderCF = ...,
    *,
    like: ArrayLike = ...,
) -> NDArray[float64]: ...
@overload
def zeros(
    shape: _ShapeLike,
    dtype: _DTypeLike[_SCT],
    order: _OrderCF = ...,
    *,
    like: ArrayLike = ...,
) -> NDArray[_SCT]: ...
@overload
def zeros(
    shape: _ShapeLike, dtype: DTypeLike, order: _OrderCF = ..., *, like: ArrayLike = ...
) -> NDArray[Any]: ...
@overload
def empty(
    shape: _ShapeLike,
    dtype: None = ...,
    order: _OrderCF = ...,
    *,
    like: ArrayLike = ...,
) -> NDArray[float64]: ...
@overload
def empty(
    shape: _ShapeLike,
    dtype: _DTypeLike[_SCT],
    order: _OrderCF = ...,
    *,
    like: ArrayLike = ...,
) -> NDArray[_SCT]: ...
@overload
def empty(
    shape: _ShapeLike, dtype: DTypeLike, order: _OrderCF = ..., *, like: ArrayLike = ...
) -> NDArray[Any]: ...
@overload
def unravel_index(
    indices: _IntLike_co, shape: _ShapeLike, order: _OrderCF = ...
) -> Tuple[intp, ...]: ...
@overload
def unravel_index(
    indices: _ArrayLikeInt_co, shape: _ShapeLike, order: _OrderCF = ...
) -> Tuple[NDArray[intp], ...]: ...
@overload
def ravel_multi_index(
    multi_index: Sequence[_IntLike_co],
    dims: Sequence[SupportsIndex],
    mode: Union[_ModeKind, Tuple[_ModeKind, ...]] = ...,
    order: _OrderCF = ...,
) -> intp: ...
@overload
def ravel_multi_index(
    multi_index: Sequence[_ArrayLikeInt_co],
    dims: Sequence[SupportsIndex],
    mode: Union[_ModeKind, Tuple[_ModeKind, ...]] = ...,
    order: _OrderCF = ...,
) -> NDArray[intp]: ...
@overload
def concatenate(
    arrays: _ArrayLike[_SCT],
    /,
    axis: Optional[SupportsIndex] = ...,
    out: None = ...,
    *,
    dtype: None = ...,
    casting: Optional[_CastingKind] = ...,
) -> NDArray[_SCT]: ...
@overload
def concatenate(
    arrays: ArrayLike,
    /,
    axis: Optional[SupportsIndex] = ...,
    out: None = ...,
    *,
    dtype: None = ...,
    casting: Optional[_CastingKind] = ...,
) -> NDArray[Any]: ...
@overload
def concatenate(
    arrays: ArrayLike,
    /,
    axis: Optional[SupportsIndex] = ...,
    out: None = ...,
    *,
    dtype: _DTypeLike[_SCT],
    casting: Optional[_CastingKind] = ...,
) -> NDArray[_SCT]: ...
@overload
def concatenate(
    arrays: ArrayLike,
    /,
    axis: Optional[SupportsIndex] = ...,
    out: None = ...,
    *,
    dtype: DTypeLike,
    casting: Optional[_CastingKind] = ...,
) -> NDArray[Any]: ...
@overload
def concatenate(
    arrays: ArrayLike,
    /,
    axis: Optional[SupportsIndex] = ...,
    out: _ArrayType = ...,
    *,
    dtype: DTypeLike = ...,
    casting: Optional[_CastingKind] = ...,
) -> _ArrayType: ...
def inner(a: ArrayLike, b: ArrayLike, /) -> Any: ...
@overload
def where(condition: ArrayLike, /) -> Tuple[NDArray[intp], ...]: ...
@overload
def where(condition: ArrayLike, x: ArrayLike, y: ArrayLike, /) -> NDArray[Any]: ...
def lexsort(keys: ArrayLike, axis: Optional[SupportsIndex] = ...) -> Any: ...
def can_cast(
    from_: Union[ArrayLike, DTypeLike],
    to: DTypeLike,
    casting: Optional[_CastingKind] = ...,
) -> bool: ...
def min_scalar_type(a: ArrayLike, /) -> dtype[Any]: ...
def result_type(*arrays_and_dtypes: Union[ArrayLike, DTypeLike]) -> dtype[Any]: ...
@overload
def dot(a: ArrayLike, b: ArrayLike, out: None = ...) -> Any: ...
@overload
def dot(a: ArrayLike, b: ArrayLike, out: _ArrayType) -> _ArrayType: ...
@overload
def vdot(a: _ArrayLikeBool_co, b: _ArrayLikeBool_co, /) -> bool_: ...
@overload
def vdot(a: _ArrayLikeUInt_co, b: _ArrayLikeUInt_co, /) -> unsignedinteger[Any]: ...
@overload
def vdot(a: _ArrayLikeInt_co, b: _ArrayLikeInt_co, /) -> signedinteger[Any]: ...
@overload
def vdot(a: _ArrayLikeFloat_co, b: _ArrayLikeFloat_co, /) -> floating[Any]: ...
@overload
def vdot(
    a: _ArrayLikeComplex_co, b: _ArrayLikeComplex_co, /
) -> complexfloating[Any, Any]: ...
@overload
def vdot(a: _ArrayLikeTD64_co, b: _ArrayLikeTD64_co, /) -> timedelta64: ...
@overload
def vdot(a: _ArrayLikeObject_co, b: Any, /) -> Any: ...
@overload
def vdot(a: Any, b: _ArrayLikeObject_co, /) -> Any: ...
def bincount(
    x: ArrayLike, /, weights: Optional[ArrayLike] = ..., minlength: SupportsIndex = ...
) -> NDArray[intp]: ...
def copyto(
    dst: NDArray[Any],
    src: ArrayLike,
    casting: Optional[_CastingKind] = ...,
    where: Optional[_ArrayLikeBool_co] = ...,
) -> None: ...
def putmask(a: NDArray[Any], mask: _ArrayLikeBool_co, values: ArrayLike) -> None: ...
def packbits(
    a: _ArrayLikeInt_co,
    /,
    axis: Optional[SupportsIndex] = ...,
    bitorder: L["big", "little"] = ...,
) -> NDArray[uint8]: ...
def unpackbits(
    a: _ArrayLike[uint8],
    /,
    axis: Optional[SupportsIndex] = ...,
    count: Optional[SupportsIndex] = ...,
    bitorder: L["big", "little"] = ...,
) -> NDArray[uint8]: ...
def shares_memory(a: object, b: object, /, max_work: Optional[int] = ...) -> bool: ...
def may_share_memory(
    a: object, b: object, /, max_work: Optional[int] = ...
) -> bool: ...
@overload
def asarray(
    a: _ArrayLike[_SCT],
    dtype: None = ...,
    order: _OrderKACF = ...,
    *,
    like: ArrayLike = ...,
) -> NDArray[_SCT]: ...
@overload
def asarray(
    a: object, dtype: None = ..., order: _OrderKACF = ..., *, like: ArrayLike = ...
) -> NDArray[Any]: ...
@overload
def asarray(
    a: Any, dtype: _DTypeLike[_SCT], order: _OrderKACF = ..., *, like: ArrayLike = ...
) -> NDArray[_SCT]: ...
@overload
def asarray(
    a: Any, dtype: DTypeLike, order: _OrderKACF = ..., *, like: ArrayLike = ...
) -> NDArray[Any]: ...
@overload
def asanyarray(
    a: _ArrayType, dtype: None = ..., order: _OrderKACF = ..., *, like: ArrayLike = ...
) -> _ArrayType: ...
@overload
def asanyarray(
    a: _ArrayLike[_SCT],
    dtype: None = ...,
    order: _OrderKACF = ...,
    *,
    like: ArrayLike = ...,
) -> NDArray[_SCT]: ...
@overload
def asanyarray(
    a: object, dtype: None = ..., order: _OrderKACF = ..., *, like: ArrayLike = ...
) -> NDArray[Any]: ...
@overload
def asanyarray(
    a: Any, dtype: _DTypeLike[_SCT], order: _OrderKACF = ..., *, like: ArrayLike = ...
) -> NDArray[_SCT]: ...
@overload
def asanyarray(
    a: Any, dtype: DTypeLike, order: _OrderKACF = ..., *, like: ArrayLike = ...
) -> NDArray[Any]: ...
@overload
def ascontiguousarray(
    a: _ArrayLike[_SCT], dtype: None = ..., *, like: ArrayLike = ...
) -> NDArray[_SCT]: ...
@overload
def ascontiguousarray(
    a: object, dtype: None = ..., *, like: ArrayLike = ...
) -> NDArray[Any]: ...
@overload
def ascontiguousarray(
    a: Any, dtype: _DTypeLike[_SCT], *, like: ArrayLike = ...
) -> NDArray[_SCT]: ...
@overload
def ascontiguousarray(
    a: Any, dtype: DTypeLike, *, like: ArrayLike = ...
) -> NDArray[Any]: ...
@overload
def asfortranarray(
    a: _ArrayLike[_SCT], dtype: None = ..., *, like: ArrayLike = ...
) -> NDArray[_SCT]: ...
@overload
def asfortranarray(
    a: object, dtype: None = ..., *, like: ArrayLike = ...
) -> NDArray[Any]: ...
@overload
def asfortranarray(
    a: Any, dtype: _DTypeLike[_SCT], *, like: ArrayLike = ...
) -> NDArray[_SCT]: ...
@overload
def asfortranarray(
    a: Any, dtype: DTypeLike, *, like: ArrayLike = ...
) -> NDArray[Any]: ...
def geterrobj() -> List[Any]: ...
def seterrobj(errobj: List[Any], /) -> None: ...
def promote_types(__type1: DTypeLike, __type2: DTypeLike) -> dtype[Any]: ...
@overload
def fromstring(
    string: str | bytes,
    dtype: None = ...,
    count: SupportsIndex = ...,
    *,
    sep: str,
    like: ArrayLike = ...,
) -> NDArray[float64]: ...
@overload
def fromstring(
    string: str | bytes,
    dtype: _DTypeLike[_SCT],
    count: SupportsIndex = ...,
    *,
    sep: str,
    like: ArrayLike = ...,
) -> NDArray[_SCT]: ...
@overload
def fromstring(
    string: str | bytes,
    dtype: DTypeLike,
    count: SupportsIndex = ...,
    *,
    sep: str,
    like: ArrayLike = ...,
) -> NDArray[Any]: ...
def frompyfunc(
    func: Callable[..., Any],
    /,
    nin: SupportsIndex,
    nout: SupportsIndex,
    *,
    identity: Any = ...,
) -> ufunc: ...
@overload
def fromfile(
    file: str | bytes | os.PathLike[Any] | _IOProtocol,
    dtype: None = ...,
    count: SupportsIndex = ...,
    sep: str = ...,
    offset: SupportsIndex = ...,
    *,
    like: ArrayLike = ...,
) -> NDArray[float64]: ...
@overload
def fromfile(
    file: str | bytes | os.PathLike[Any] | _IOProtocol,
    dtype: _DTypeLike[_SCT],
    count: SupportsIndex = ...,
    sep: str = ...,
    offset: SupportsIndex = ...,
    *,
    like: ArrayLike = ...,
) -> NDArray[_SCT]: ...
@overload
def fromfile(
    file: str | bytes | os.PathLike[Any] | _IOProtocol,
    dtype: DTypeLike,
    count: SupportsIndex = ...,
    sep: str = ...,
    offset: SupportsIndex = ...,
    *,
    like: ArrayLike = ...,
) -> NDArray[Any]: ...
@overload
def fromiter(
    iter: Iterable[Any],
    dtype: _DTypeLike[_SCT],
    count: SupportsIndex = ...,
    *,
    like: ArrayLike = ...,
) -> NDArray[_SCT]: ...
@overload
def fromiter(
    iter: Iterable[Any],
    dtype: DTypeLike,
    count: SupportsIndex = ...,
    *,
    like: ArrayLike = ...,
) -> NDArray[Any]: ...
@overload
def frombuffer(
    buffer: _SupportsBuffer,
    dtype: None = ...,
    count: SupportsIndex = ...,
    offset: SupportsIndex = ...,
    *,
    like: ArrayLike = ...,
) -> NDArray[float64]: ...
@overload
def frombuffer(
    buffer: _SupportsBuffer,
    dtype: _DTypeLike[_SCT],
    count: SupportsIndex = ...,
    offset: SupportsIndex = ...,
    *,
    like: ArrayLike = ...,
) -> NDArray[_SCT]: ...
@overload
def frombuffer(
    buffer: _SupportsBuffer,
    dtype: DTypeLike,
    count: SupportsIndex = ...,
    offset: SupportsIndex = ...,
    *,
    like: ArrayLike = ...,
) -> NDArray[Any]: ...
@overload
def arange(
    stop: _IntLike_co, /, *, dtype: None = ..., like: ArrayLike = ...
) -> NDArray[signedinteger[Any]]: ...
@overload
def arange(
    start: _IntLike_co,
    stop: _IntLike_co,
    step: _IntLike_co = ...,
    dtype: None = ...,
    *,
    like: ArrayLike = ...,
) -> NDArray[signedinteger[Any]]: ...
@overload
def arange(
    stop: _FloatLike_co, /, *, dtype: None = ..., like: ArrayLike = ...
) -> NDArray[floating[Any]]: ...
@overload
def arange(
    start: _FloatLike_co,
    stop: _FloatLike_co,
    step: _FloatLike_co = ...,
    dtype: None = ...,
    *,
    like: ArrayLike = ...,
) -> NDArray[floating[Any]]: ...
@overload
def arange(
    stop: _TD64Like_co, /, *, dtype: None = ..., like: ArrayLike = ...
) -> NDArray[timedelta64]: ...
@overload
def arange(
    start: _TD64Like_co,
    stop: _TD64Like_co,
    step: _TD64Like_co = ...,
    dtype: None = ...,
    *,
    like: ArrayLike = ...,
) -> NDArray[timedelta64]: ...
@overload
def arange(
    start: datetime64,
    stop: datetime64,
    step: datetime64 = ...,
    dtype: None = ...,
    *,
    like: ArrayLike = ...,
) -> NDArray[datetime64]: ...
@overload
def arange(
    stop: Any, /, *, dtype: _DTypeLike[_SCT], like: ArrayLike = ...
) -> NDArray[_SCT]: ...
@overload
def arange(
    start: Any,
    stop: Any,
    step: Any = ...,
    dtype: _DTypeLike[_SCT] = ...,
    *,
    like: ArrayLike = ...,
) -> NDArray[_SCT]: ...
@overload
def arange(
    stop: Any, /, *, dtype: DTypeLike, like: ArrayLike = ...
) -> NDArray[Any]: ...
@overload
def arange(
    start: Any,
    stop: Any,
    step: Any = ...,
    dtype: DTypeLike = ...,
    *,
    like: ArrayLike = ...,
) -> NDArray[Any]: ...
def datetime_data(
    dtype: str | _DTypeLike[datetime64] | _DTypeLike[timedelta64], /
) -> Tuple[str, int]: ...
@overload
def busday_count(
    begindates: _ScalarLike_co,
    enddates: _ScalarLike_co,
    weekmask: ArrayLike = ...,
    holidays: None | ArrayLike = ...,
    busdaycal: None | busdaycalendar = ...,
    out: None = ...,
) -> int_: ...
@overload
def busday_count(
    begindates: ArrayLike,
    enddates: ArrayLike,
    weekmask: ArrayLike = ...,
    holidays: None | ArrayLike = ...,
    busdaycal: None | busdaycalendar = ...,
    out: None = ...,
) -> NDArray[int_]: ...
@overload
def busday_count(
    begindates: ArrayLike,
    enddates: ArrayLike,
    weekmask: ArrayLike = ...,
    holidays: None | ArrayLike = ...,
    busdaycal: None | busdaycalendar = ...,
    out: _ArrayType = ...,
) -> _ArrayType: ...
@overload
def busday_offset(
    dates: datetime64,
    offsets: _TD64Like_co,
    roll: L["raise"] = ...,
    weekmask: ArrayLike = ...,
    holidays: None | ArrayLike = ...,
    busdaycal: None | busdaycalendar = ...,
    out: None = ...,
) -> datetime64: ...
@overload
def busday_offset(
    dates: _ArrayLike[datetime64],
    offsets: _ArrayLikeTD64_co,
    roll: L["raise"] = ...,
    weekmask: ArrayLike = ...,
    holidays: None | ArrayLike = ...,
    busdaycal: None | busdaycalendar = ...,
    out: None = ...,
) -> NDArray[datetime64]: ...
@overload
def busday_offset(
    dates: _ArrayLike[datetime64],
    offsets: _ArrayLike[timedelta64],
    roll: L["raise"] = ...,
    weekmask: ArrayLike = ...,
    holidays: None | ArrayLike = ...,
    busdaycal: None | busdaycalendar = ...,
    out: _ArrayType = ...,
) -> _ArrayType: ...
@overload
def busday_offset(
    dates: _ScalarLike_co,
    offsets: _ScalarLike_co,
    roll: _RollKind,
    weekmask: ArrayLike = ...,
    holidays: None | ArrayLike = ...,
    busdaycal: None | busdaycalendar = ...,
    out: None = ...,
) -> datetime64: ...
@overload
def busday_offset(
    dates: ArrayLike,
    offsets: ArrayLike,
    roll: _RollKind,
    weekmask: ArrayLike = ...,
    holidays: None | ArrayLike = ...,
    busdaycal: None | busdaycalendar = ...,
    out: None = ...,
) -> NDArray[datetime64]: ...
@overload
def busday_offset(
    dates: ArrayLike,
    offsets: ArrayLike,
    roll: _RollKind,
    weekmask: ArrayLike = ...,
    holidays: None | ArrayLike = ...,
    busdaycal: None | busdaycalendar = ...,
    out: _ArrayType = ...,
) -> _ArrayType: ...
@overload
def is_busday(
    dates: _ScalarLike_co,
    weekmask: ArrayLike = ...,
    holidays: None | ArrayLike = ...,
    busdaycal: None | busdaycalendar = ...,
    out: None = ...,
) -> bool_: ...
@overload
def is_busday(
    dates: ArrayLike,
    weekmask: ArrayLike = ...,
    holidays: None | ArrayLike = ...,
    busdaycal: None | busdaycalendar = ...,
    out: None = ...,
) -> NDArray[bool_]: ...
@overload
def is_busday(
    dates: ArrayLike,
    weekmask: ArrayLike = ...,
    holidays: None | ArrayLike = ...,
    busdaycal: None | busdaycalendar = ...,
    out: _ArrayType = ...,
) -> _ArrayType: ...
@overload
def datetime_as_string(
    arr: datetime64,
    unit: None | L["auto"] | _UnitKind = ...,
    timezone: L["naive", "UTC", "local"] | dt.tzinfo = ...,
    casting: _CastingKind = ...,
) -> str_: ...
@overload
def datetime_as_string(
    arr: _ArrayLikeDT64_co,
    unit: None | L["auto"] | _UnitKind = ...,
    timezone: L["naive", "UTC", "local"] | dt.tzinfo = ...,
    casting: _CastingKind = ...,
) -> NDArray[str_]: ...
@overload
def compare_chararrays(
    a1: _ArrayLikeStr_co,
    a2: _ArrayLikeStr_co,
    cmp: L["<", "<=", "==", ">=", ">", "!="],
    rstrip: bool,
) -> NDArray[bool_]: ...
@overload
def compare_chararrays(
    a1: _ArrayLikeBytes_co,
    a2: _ArrayLikeBytes_co,
    cmp: L["<", "<=", "==", ">=", ">", "!="],
    rstrip: bool,
) -> NDArray[bool_]: ...
def add_docstring(obj: Callable[..., Any], docstring: str, /) -> None: ...

_GetItemKeys = L[
    "C",
    "CONTIGUOUS",
    "C_CONTIGUOUS",
    "F",
    "FORTRAN",
    "F_CONTIGUOUS",
    "W",
    "WRITEABLE",
    "B",
    "BEHAVED",
    "O",
    "OWNDATA",
    "A",
    "ALIGNED",
    "X",
    "WRITEBACKIFCOPY",
    "CA",
    "CARRAY",
    "FA",
    "FARRAY",
    "FNC",
    "FORC",
]
_SetItemKeys = L["A", "ALIGNED", "W", "WRITABLE", "X", "WRITEBACKIFCOPY"]

@final
class flagsobj:
    __hash__: None
    aligned: bool
    writeable: bool
    writebackifcopy: bool
    @property
    def behaved(self) -> bool: ...
    @property
    def c_contiguous(self) -> bool: ...
    @property
    def carray(self) -> bool: ...
    @property
    def contiguous(self) -> bool: ...
    @property
    def f_contiguous(self) -> bool: ...
    @property
    def farray(self) -> bool: ...
    @property
    def fnc(self) -> bool: ...
    @property
    def forc(self) -> bool: ...
    @property
    def fortran(self) -> bool: ...
    @property
    def num(self) -> int: ...
    @property
    def owndata(self) -> bool: ...
    def __getitem__(self, key: _GetItemKeys) -> bool: ...
    def __setitem__(self, key: _SetItemKeys, value: bool) -> None: ...

def nested_iters(
    op: ArrayLike | Sequence[ArrayLike],
    axes: Sequence[Sequence[SupportsIndex]],
    flags: None | Sequence[_NDIterFlagsKind] = ...,
    op_flags: None | Sequence[Sequence[_NDIterOpFlagsKind]] = ...,
    op_dtypes: DTypeLike | Sequence[DTypeLike] = ...,
    order: _OrderKACF = ...,
    casting: _CastingKind = ...,
    buffersize: SupportsIndex = ...,
) -> Tuple[nditer, ...]: ...
