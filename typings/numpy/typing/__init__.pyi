import sys
from typing import TYPE_CHECKING, List

from numpy import ufunc
from numpy._pytesttester import PytestTester

from ._add_docstring import _docstrings
from ._array_like import ArrayLike as ArrayLike
from ._array_like import (
    _ArrayLike,
    _ArrayLikeBool_co,
    _ArrayLikeBytes_co,
    _ArrayLikeComplex_co,
    _ArrayLikeDT64_co,
    _ArrayLikeFloat_co,
    _ArrayLikeInt,
    _ArrayLikeInt_co,
    _ArrayLikeNumber_co,
    _ArrayLikeObject_co,
    _ArrayLikeStr_co,
    _ArrayLikeTD64_co,
    _ArrayLikeUInt_co,
    _ArrayLikeVoid_co,
    _NestedSequence,
    _RecursiveSequence,
    _SupportsArray,
)
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
from ._dtype_like import DTypeLike as DTypeLike
from ._dtype_like import (
    _DTypeLikeBool,
    _DTypeLikeBytes,
    _DTypeLikeComplex,
    _DTypeLikeComplex_co,
    _DTypeLikeDT64,
    _DTypeLikeFloat,
    _DTypeLikeInt,
    _DTypeLikeObject,
    _DTypeLikeStr,
    _DTypeLikeTD64,
    _DTypeLikeUInt,
    _DTypeLikeVoid,
    _SupportsDType,
    _VoidDTypeLike,
)
from ._generic_alias import NDArray as NDArray
from ._generic_alias import _GenericAlias
from ._nbit import (
    _NBitByte,
    _NBitDouble,
    _NBitHalf,
    _NBitInt,
    _NBitIntC,
    _NBitIntP,
    _NBitLongDouble,
    _NBitLongLong,
    _NBitShort,
    _NBitSingle,
)
from ._scalars import (
    _BoolLike_co,
    _CharLike_co,
    _ComplexLike_co,
    _FloatLike_co,
    _IntLike_co,
    _NumberLike_co,
    _ScalarLike_co,
    _TD64Like_co,
    _UIntLike_co,
    _VoidLike_co,
)
from ._shape import _Shape, _ShapeLike
from ._ufunc import (
    _GUFunc_Nin2_Nout1,
    _UFunc_Nin1_Nout1,
    _UFunc_Nin1_Nout2,
    _UFunc_Nin2_Nout1,
    _UFunc_Nin2_Nout2,
)

"""
============================
Typing (:mod:`numpy.typing`)
============================

.. warning::

  Some of the types in this module rely on features only present in
  the standard library in Python 3.8 and greater. If you want to use
  these types in earlier versions of Python, you should install the
  typing-extensions_ package.

Large parts of the NumPy API have PEP-484-style type annotations. In
addition a number of type aliases are available to users, most prominently
the two below:

- `ArrayLike`: objects that can be converted to arrays
- `DTypeLike`: objects that can be converted to dtypes

.. _typing-extensions: https://pypi.org/project/typing-extensions/

Mypy plugin
-----------

A mypy_ plugin is distributed in `numpy.typing` for managing a number of
platform-specific annotations. Its function can be split into to parts:

* Assigning the (platform-dependent) precisions of certain `~numpy.number` subclasses,
  including the likes of `~numpy.int_`, `~numpy.intp` and `~numpy.longlong`.
  See the documentation on :ref:`scalar types <arrays.scalars.built-in>` for a
  comprehensive overview of the affected classes. without the plugin the precision
  of all relevant classes will be inferred as `~typing.Any`.
* Removing all extended-precision `~numpy.number` subclasses that are unavailable
  for the platform in question. Most notable this includes the likes of
  `~numpy.float128` and `~numpy.complex256`. Without the plugin *all*
  extended-precision types will, as far as mypy is concerned, be available
  to all platforms.

To enable the plugin, one must add it to their mypy `configuration file`_:

.. code-block:: ini

    [mypy]
    plugins = numpy.typing.mypy_plugin

.. _mypy: http://mypy-lang.org/
.. _configuration file: https://mypy.readthedocs.io/en/stable/config_file.html

Differences from the runtime NumPy API
--------------------------------------

NumPy is very flexible. Trying to describe the full range of
possibilities statically would result in types that are not very
helpful. For that reason, the typed NumPy API is often stricter than
the runtime NumPy API. This section describes some notable
differences.

ArrayLike
~~~~~~~~~

The `ArrayLike` type tries to avoid creating object arrays. For
example,

.. code-block:: python

    >>> np.array(x**2 for x in range(10))
    array(<generator object <genexpr> at ...>, dtype=object)

is valid NumPy code which will create a 0-dimensional object
array. Type checkers will complain about the above example when using
the NumPy types however. If you really intended to do the above, then
you can either use a ``# type: ignore`` comment:

.. code-block:: python

    >>> np.array(x**2 for x in range(10))  # type: ignore

or explicitly type the array like object as `~typing.Any`:

.. code-block:: python

    >>> from typing import Any
    >>> array_like: Any = (x**2 for x in range(10))
    >>> np.array(array_like)
    array(<generator object <genexpr> at ...>, dtype=object)

ndarray
~~~~~~~

It's possible to mutate the dtype of an array at runtime. For example,
the following code is valid:

.. code-block:: python

    >>> x = np.array([1, 2])
    >>> x.dtype = np.bool_

This sort of mutation is not allowed by the types. Users who want to
write statically typed code should instead use the `numpy.ndarray.view`
method to create a view of the array with a different dtype.

DTypeLike
~~~~~~~~~

The `DTypeLike` type tries to avoid creation of dtype objects using
dictionary of fields like below:

.. code-block:: python

    >>> x = np.dtype({"field1": (float, 1), "field2": (int, 3)})

Although this is valid NumPy code, the type checker will complain about it,
since its usage is discouraged.
Please see : :ref:`Data type objects <arrays.dtypes>`

Number precision
~~~~~~~~~~~~~~~~

The precision of `numpy.number` subclasses is treated as a covariant generic
parameter (see :class:`~NBitBase`), simplifying the annotating of processes
involving precision-based casting.

.. code-block:: python

    >>> from typing import TypeVar
    >>> import numpy as np
    >>> import numpy.typing as npt

    >>> T = TypeVar("T", bound=npt.NBitBase)
    >>> def func(a: "np.floating[T]", b: "np.floating[T]") -> "np.floating[T]":
    ...     ...

Consequently, the likes of `~numpy.float16`, `~numpy.float32` and
`~numpy.float64` are still sub-types of `~numpy.floating`, but, contrary to
runtime, they're not necessarily considered as sub-classes.

Timedelta64
~~~~~~~~~~~

The `~numpy.timedelta64` class is not considered a subclass of `~numpy.signedinteger`,
the former only inheriting from `~numpy.generic` while static type checking.

0D arrays
~~~~~~~~~

During runtime numpy aggressively casts any passed 0D arrays into their
corresponding `~numpy.generic` instance. Until the introduction of shape
typing (see :pep:`646`) it is unfortunately not possible to make the
necessary distinction between 0D and >0D arrays. While thus not strictly
correct, all operations are that can potentially perform a 0D-array -> scalar
cast are currently annotated as exclusively returning an `ndarray`.

If it is known in advance that an operation _will_ perform a
0D-array -> scalar cast, then one can consider manually remedying the
situation with either `typing.cast` or a ``# type: ignore`` comment.

API
---

"""
if TYPE_CHECKING: ...
else: ...
if notTYPE_CHECKING: ...
else:
    __all__: List[str]
    ...

@final
class NBitBase:
    """
    An object representing `numpy.number` precision during static type checking.

    Used exclusively for the purpose static type checking, `NBitBase`
    represents the base of a hierarchical set of subclasses.
    Each subsequent subclass is herein used for representing a lower level
    of precision, *e.g.* ``64Bit > 32Bit > 16Bit``.

    Examples
    --------
    Below is a typical usage example: `NBitBase` is herein used for annotating a
    function that takes a float and integer of arbitrary precision as arguments
    and returns a new float of whichever precision is largest
    (*e.g.* ``np.float16 + np.int64 -> np.float64``).

    .. code-block:: python

        >>> from __future__ import annotations
        >>> from typing import TypeVar, Union, TYPE_CHECKING
        >>> import numpy as np
        >>> import numpy.typing as npt

        >>> T1 = TypeVar("T1", bound=npt.NBitBase)
        >>> T2 = TypeVar("T2", bound=npt.NBitBase)

        >>> def add(a: np.floating[T1], b: np.integer[T2]) -> np.floating[Union[T1, T2]]:
        ...     return a + b

        >>> a = np.float16()
        >>> b = np.int64()
        >>> out = add(a, b)

        >>> if TYPE_CHECKING:
        ...     reveal_locals()
        ...     # note: Revealed local types are:
        ...     # note:     a: numpy.floating[numpy.typing._16Bit*]
        ...     # note:     b: numpy.signedinteger[numpy.typing._64Bit*]
        ...     # note:     out: numpy.floating[numpy.typing._64Bit*]

    """

    def __init_subclass__(cls) -> None: ...

class _256Bit(NBitBase): ...
class _128Bit(_256Bit): ...
class _96Bit(_128Bit): ...
class _80Bit(_96Bit): ...
class _64Bit(_80Bit): ...
class _32Bit(_64Bit): ...
class _16Bit(_32Bit): ...
class _8Bit(_16Bit): ...

if TYPE_CHECKING: ...
else: ...
if __doc__ is not None: ...
test = ...
