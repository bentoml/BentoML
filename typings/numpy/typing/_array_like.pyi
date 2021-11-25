import sys
from typing import TYPE_CHECKING, Any, Protocol, Sequence, Union

from numpy import dtype, ndarray

if sys.version_info >= (3, 8):
    HAVE_PROTOCOL = ...
else: ...
_T = ...
_ScalarType = ...
_DType = ...
_DType_co = ...
if TYPE_CHECKING or HAVE_PROTOCOL:
    class _SupportsArray(Protocol[_DType_co]):
        def __array__(self) -> ndarray[Any, _DType_co]: ...

else: ...
_NestedSequence = (
    Union[
        _T,
        Sequence[_T],
        Sequence[Sequence[_T]],
        Sequence[Sequence[Sequence[_T]]],
        Sequence[Sequence[Sequence[Sequence[_T]]]],
    ],
)
_RecursiveSequence = Sequence[Sequence[Sequence[Sequence[Sequence[Any]]]]]
_ArrayLike = (Union[_NestedSequence[_SupportsArray[_DType]], _NestedSequence[_T]],)
ArrayLike = (
    Union[
        _RecursiveSequence,
        _ArrayLike[dtype, Union[bool, int, float, complex, str, bytes]],
    ],
)
_ArrayLikeBool_co = (_ArrayLike["dtype[bool_]", bool],)
_ArrayLikeUInt_co = (_ArrayLike["dtype[Union[bool_, unsignedinteger[Any]]]", bool],)
_ArrayLikeInt_co = (_ArrayLike["dtype[Union[bool_, integer[Any]]]", Union[bool, int]],)
_ArrayLikeFloat_co = (
    _ArrayLike[
        "dtype[Union[bool_, integer[Any], floating[Any]]]", Union[bool, int, float]
    ],
)
_ArrayLikeComplex_co = (
    _ArrayLike[
        "dtype[Union[bool_, integer[Any], floating[Any], complexfloating[Any, Any]]]",
        Union[bool, int, float, complex],
    ],
)
_ArrayLikeNumber_co = (
    _ArrayLike["dtype[Union[bool_, number[Any]]]", Union[bool, int, float, complex]],
)
_ArrayLikeTD64_co = (
    _ArrayLike["dtype[Union[bool_, integer[Any], timedelta64]]", Union[bool, int]],
)
_ArrayLikeDT64_co = _NestedSequence[_SupportsArray["dtype[datetime64]"]]
_ArrayLikeObject_co = _NestedSequence[_SupportsArray["dtype[object_]"]]
_ArrayLikeVoid_co = _NestedSequence[_SupportsArray["dtype[void]"]]
_ArrayLikeStr_co = (_ArrayLike["dtype[str_]", str],)
_ArrayLikeBytes_co = (_ArrayLike["dtype[bytes_]", bytes],)
_ArrayLikeInt = (_ArrayLike["dtype[integer[Any]]", int],)
