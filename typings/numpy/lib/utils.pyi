from ast import AST
from typing import (
    Any,
    Callable,
    List,
    Mapping,
    Optional,
    Protocol,
    Sequence,
    Tuple,
    TypeVar,
    Union,
    overload,
)
from numpy import generic, ndarray
from numpy.core.numerictypes import issubclass_ as issubclass_
from numpy.core.numerictypes import issubdtype as issubdtype
from numpy.core.numerictypes import issubsctype as issubsctype

_T_contra = TypeVar("_T_contra", contravariant=True)
_FuncType = TypeVar("_FuncType", bound=Callable[..., Any])

class _SupportsWrite(Protocol[_T_contra]):
    def write(self, s: _T_contra, /) -> Any: ...

__all__: List[str]

class _Deprecate:
    old_name: Optional[str]
    new_name: Optional[str]
    message: Optional[str]
    def __init__(
        self,
        old_name: Optional[str] = ...,
        new_name: Optional[str] = ...,
        message: Optional[str] = ...,
    ) -> None: ...
    def __call__(self, func: _FuncType) -> _FuncType: ...

def get_include() -> str: ...
@overload
def deprecate(
    *,
    old_name: Optional[str] = ...,
    new_name: Optional[str] = ...,
    message: Optional[str] = ...,
) -> _Deprecate: ...
@overload
def deprecate(
    func: _FuncType,
    /,
    old_name: Optional[str] = ...,
    new_name: Optional[str] = ...,
    message: Optional[str] = ...,
) -> _FuncType: ...
def deprecate_with_doc(msg: Optional[str]) -> _Deprecate: ...
def byte_bounds(a: Union[generic, ndarray[Any, Any]]) -> Tuple[int, int]: ...
def who(vardict: Optional[Mapping[str, ndarray[Any, Any]]] = ...) -> None: ...
def info(
    object: object = ...,
    maxwidth: int = ...,
    output: Optional[_SupportsWrite[str]] = ...,
    toplevel: str = ...,
) -> None: ...
def source(object: object, output: Optional[_SupportsWrite[str]] = ...) -> None: ...
def lookfor(
    what: str,
    module: Union[None, str, Sequence[str]] = ...,
    import_modules: bool = ...,
    regenerate: bool = ...,
    output: Optional[_SupportsWrite[str]] = ...,
) -> None: ...
def safe_eval(source: Union[str, AST]) -> Any: ...
