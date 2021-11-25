import sys
from typing import (
    TYPE_CHECKING,
    Any,
    List,
    Optional,
    Protocol,
    Sequence,
    Tuple,
    Type,
    TypedDict,
    TypeVar,
    Union,
)

import numpy as np

from ._char_codes import (
    _BoolCodes,
    _ByteCodes,
    _BytesCodes,
    _CDoubleCodes,
    _CLongDoubleCodes,
    _Complex64Codes,
    _Complex128Codes,
    _CSingleCodes,
    _DoubleCodes,
    _DT64Codes,
    _Float16Codes,
    _Float32Codes,
    _Float64Codes,
    _HalfCodes,
    _Int8Codes,
    _Int16Codes,
    _Int32Codes,
    _Int64Codes,
    _IntCCodes,
    _IntCodes,
    _IntPCodes,
    _LongDoubleCodes,
    _LongLongCodes,
    _ObjectCodes,
    _ShortCodes,
    _SingleCodes,
    _StrCodes,
    _TD64Codes,
    _UByteCodes,
    _UInt8Codes,
    _UInt16Codes,
    _UInt32Codes,
    _UInt64Codes,
    _UIntCCodes,
    _UIntCodes,
    _UIntPCodes,
    _ULongLongCodes,
    _UShortCodes,
    _VoidCodes,
)
from ._shape import _ShapeLike

_DType_co = TypeVar("_DType_co", covariant=True)
_DTypeLikeNested = Any

class _DTypeDictBase(TypedDict):
    names: Sequence[str]
    formats: Sequence[_DTypeLikeNested]
    ...

class _DTypeDict(_DTypeDictBase, total=False):
    offsets: Sequence[int]
    titles: Sequence[Any]
    itemsize: int
    aligned: bool
    ...

class _SupportsDType(Protocol[_DType_co]):
    @property
    def dtype(self) -> _DType_co: ...

_VoidDTypeLike = Union[
    Tuple[_DTypeLikeNested, int],
    Tuple[_DTypeLikeNested, _ShapeLike],
    List[Any],
    _DTypeDict,
    Tuple[_DTypeLikeNested, _DTypeLikeNested],
]

DTypeLike = Optional[
    Union[np.dtype[Any], type, _SupportsDType[np.dtype[Any]], str, _VoidDTypeLike]
]
_DTypeLikeBool = Union[
    Type[bool],
    Type[np.bool_],
    "np.dtype[np.bool_]",
    "_SupportsDType[np.dtype[np.bool_]]",
    _BoolCodes,
]

_DTypeLikeUInt = Union[
    Type[np.unsignedinteger],
    "np.dtype[np.unsignedinteger]",
    "_SupportsDType[np.dtype[np.unsignedinteger]]",
    _UInt8Codes,
    _UInt16Codes,
    _UInt32Codes,
    _UInt64Codes,
    _UByteCodes,
    _UShortCodes,
    _UIntCCodes,
    _UIntPCodes,
    _UIntCodes,
    _ULongLongCodes,
]
_DTypeLikeInt = (
    Union[
        Type[int],
        Type[np.signedinteger],
        "np.dtype[np.signedinteger]",
        "_SupportsDType[np.dtype[np.signedinteger]]",
        _Int8Codes,
        _Int16Codes,
        _Int32Codes,
        _Int64Codes,
        _ByteCodes,
        _ShortCodes,
        _IntCCodes,
        _IntPCodes,
        _IntCodes,
        _LongLongCodes,
    ],
)
_DTypeLikeFloat = (
    Union[
        Type[float],
        Type[np.floating],
        "np.dtype[np.floating]",
        "_SupportsDType[np.dtype[np.floating]]",
        _Float16Codes,
        _Float32Codes,
        _Float64Codes,
        _HalfCodes,
        _SingleCodes,
        _DoubleCodes,
        _LongDoubleCodes,
    ],
)
_DTypeLikeComplex = (
    Union[
        Type[complex],
        Type[np.complexfloating],
        "np.dtype[np.complexfloating]",
        "_SupportsDType[np.dtype[np.complexfloating]]",
        _Complex64Codes,
        _Complex128Codes,
        _CSingleCodes,
        _CDoubleCodes,
        _CLongDoubleCodes,
    ],
)
_DTypeLikeDT64 = (
    Union[
        Type[np.timedelta64],
        "np.dtype[np.timedelta64]",
        "_SupportsDType[np.dtype[np.timedelta64]]",
        _TD64Codes,
    ],
)
_DTypeLikeTD64 = (
    Union[
        Type[np.datetime64],
        "np.dtype[np.datetime64]",
        "_SupportsDType[np.dtype[np.datetime64]]",
        _DT64Codes,
    ],
)
_DTypeLikeStr = (
    Union[
        Type[str],
        Type[np.str_],
        "np.dtype[np.str_]",
        "_SupportsDType[np.dtype[np.str_]]",
        _StrCodes,
    ],
)
_DTypeLikeBytes = (
    Union[
        Type[bytes],
        Type[np.bytes_],
        "np.dtype[np.bytes_]",
        "_SupportsDType[np.dtype[np.bytes_]]",
        _BytesCodes,
    ],
)
_DTypeLikeVoid = (
    Union[
        Type[np.void],
        "np.dtype[np.void]",
        "_SupportsDType[np.dtype[np.void]]",
        _VoidCodes,
        _VoidDTypeLike,
    ],
)
_DTypeLikeObject = (
    Union[
        type, "np.dtype[np.object_]", "_SupportsDType[np.dtype[np.object_]]", _ObjectCodes
    ],
)
_DTypeLikeComplex_co = (
    Union[
        _DTypeLikeBool, _DTypeLikeUInt, _DTypeLikeInt, _DTypeLikeFloat, _DTypeLikeComplex
    ],
)
