# Backport for numpy.typing for all numpy<1.20
import sys
from typing import TYPE_CHECKING, Any, Generic, List, Sequence, Tuple, TypeVar, Union

from ..utils.lazy_loader import LazyLoader

if TYPE_CHECKING:
    import numpy as np
else:
    np = LazyLoader("np", globals(), "numpy")


_T = TypeVar("_T")
_ScalarType = TypeVar("_ScalarType", bound="np.generic")
_DType = TypeVar("_DType", bound="np.dtype[Any]")
_DType_co = TypeVar("_DType_co", covariant=True, bound="np.dtype[Any]")
_DTypeLikeNested = Any

if sys.version_info >= (3, 8):
    from typing import Protocol, SupportsIndex, TypedDict

    HAVE_PROTOCOL = True
else:
    try:
        from typing_extensions import Protocol, SupportsIndex, TypedDict
    except ImportError:
        HAVE_PROTOCOL = False
        SupportsIndex = Any
    else:
        HAVE_PROTOCOL = True

if TYPE_CHECKING or HAVE_PROTOCOL:
    # Mandatory keys
    class _DTypeDictBase(TypedDict):
        names: Sequence[str]
        formats: Sequence[_DTypeLikeNested]

    # Mandatory + optional keys
    class _DTypeDict(_DTypeDictBase, total=False):
        offsets: Sequence[int]
        titles: Sequence[
            Any
        ]  # Only `str` elements are usable as indexing aliases, but all objects are legal
        itemsize: int
        aligned: bool

    # A protocol for anything with the dtype attribute
    class _SupportsDType(Protocol[_DType_co]):
        @property
        def dtype(self) -> _DType_co:
            ...

    # The `_SupportsArray` protocol only cares about the default dtype
    # (i.e. `dtype=None` or no `dtype` parameter at all) of the to-be returned
    # array.
    # Concrete implementations of the protocol are responsible for adding
    # any and all remaining overloads
    class _SupportsArray(Protocol[_DType_co]):
        def __array__(self) -> "np.ndarray[Any, _DType_co]":
            ...


else:
    _DTypeDict = Any

    class _SupportsDType(Generic[_DType_co]):
        ...

    class _SupportsArray(Generic[_DType_co]):
        pass


_ShapeLike = Union[SupportsIndex, Sequence[SupportsIndex]]

# Would create a dtype[np.void]
_VoidDTypeLike = Union[
    # (flexible_dtype, itemsize)
    Tuple[_DTypeLikeNested, int],
    # (fixed_dtype, shape)
    Tuple[_DTypeLikeNested, _ShapeLike],
    # [(field_name, field_dtype, field_shape), ...]
    #
    # The type here is quite broad because NumPy accepts quite a wide
    # range of inputs inside the list; see the tests for some
    # examples.
    List[Any],
    # {'names': ..., 'formats': ..., 'offsets': ..., 'titles': ...,
    #  'itemsize': ...}
    _DTypeDict,
    # (base_dtype, new_dtype)
    Tuple[_DTypeLikeNested, _DTypeLikeNested],
]

DTypeLike = Union[
    "np.dtype",
    # default data type (float64)
    None,
    # array-scalar types and generic types
    type,  # TODO: enumerate these when we add type hints for numpy scalars
    # anything with a dtype attribute
    _SupportsDType["np.dtype"],
    # character codes, type strings or comma-separated fields, e.g., 'float64'
    str,
    _VoidDTypeLike,
]


_NestedSequence = Union[
    _T,
    Sequence[_T],
    Sequence[Sequence[_T]],
    Sequence[Sequence[Sequence[_T]]],
    Sequence[Sequence[Sequence[Sequence[_T]]]],
]
_RecursiveSequence = Sequence[Sequence[Sequence[Sequence[Sequence[Any]]]]]

# A union representing array-like objects; consists of two typevars:
# One representing types that can be parametrized w.r.t. `np.dtype`
# and another one for the rest
_ArrayLike = Union[
    _NestedSequence[_SupportsArray[_DType]],
    _NestedSequence[_T],
]

# TODO: support buffer protocols once
#
# https://bugs.python.org/issue27501
#
# is resolved. See also the mypy issue:
#
# https://github.com/python/typing/issues/593
ArrayLike = Union[
    _RecursiveSequence,
    _ArrayLike["np.dtype", Union[bool, int, float, complex, str, bytes]],
]
