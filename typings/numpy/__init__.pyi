import array as _array
import builtins
import ctypes as ct
import datetime as dt
import mmap
import os
import sys
from abc import abstractmethod
from contextlib import ContextDecorator
from types import TracebackType
from typing import (
    IO,
    Any,
    ByteString,
    Callable,
    Container,
    Dict,
    Final,
    Generic,
    Iterable,
    Iterator,
    List,
)
from typing import Literal as L
from typing import (
    Mapping,
    NoReturn,
    Optional,
    Protocol,
    Sequence,
    Sized,
    SupportsComplex,
    SupportsFloat,
    SupportsIndex,
    SupportsInt,
    Text,
    Tuple,
    Type,
    TypeVar,
    Union,
    overload,
)

from numpy import char as char
from numpy import ctypeslib as ctypeslib
from numpy import fft as fft
from numpy import lib as lib
from numpy import linalg as linalg
from numpy import ma as ma
from numpy import matrixlib as matrixlib
from numpy import polynomial as polynomial
from numpy import random as random
from numpy import rec as rec
from numpy import testing as testing
from numpy import version as version
from numpy.core._asarray import asanyarray as asanyarray
from numpy.core._asarray import asarray as asarray
from numpy.core._asarray import ascontiguousarray as ascontiguousarray
from numpy.core._asarray import asfortranarray as asfortranarray
from numpy.core._asarray import require as require
from numpy.core._internal import _ctypes
from numpy.core._type_aliases import sctypeDict as sctypeDict
from numpy.core._type_aliases import sctypes as sctypes
from numpy.core._ufunc_config import (
    _ErrDictOptional,
    _ErrFunc,
    _ErrKind,
    _SupportsWrite,
)
from numpy.core._ufunc_config import getbufsize as getbufsize
from numpy.core._ufunc_config import geterr as geterr
from numpy.core._ufunc_config import geterrcall as geterrcall
from numpy.core._ufunc_config import setbufsize as setbufsize
from numpy.core._ufunc_config import seterr as seterr
from numpy.core._ufunc_config import seterrcall as seterrcall
from numpy.core.arrayprint import array2string as array2string
from numpy.core.arrayprint import array_repr as array_repr
from numpy.core.arrayprint import array_str as array_str
from numpy.core.arrayprint import format_float_positional as format_float_positional
from numpy.core.arrayprint import format_float_scientific as format_float_scientific
from numpy.core.arrayprint import get_printoptions as get_printoptions
from numpy.core.arrayprint import printoptions as printoptions
from numpy.core.arrayprint import set_printoptions as set_printoptions
from numpy.core.arrayprint import set_string_function as set_string_function
from numpy.core.einsumfunc import einsum as einsum
from numpy.core.einsumfunc import einsum_path as einsum_path
from numpy.core.fromnumeric import all as all
from numpy.core.fromnumeric import amax as amax
from numpy.core.fromnumeric import amin as amin
from numpy.core.fromnumeric import any as any
from numpy.core.fromnumeric import argmax as argmax
from numpy.core.fromnumeric import argmin as argmin
from numpy.core.fromnumeric import argpartition as argpartition
from numpy.core.fromnumeric import argsort as argsort
from numpy.core.fromnumeric import around as around
from numpy.core.fromnumeric import choose as choose
from numpy.core.fromnumeric import clip as clip
from numpy.core.fromnumeric import compress as compress
from numpy.core.fromnumeric import cumprod as cumprod
from numpy.core.fromnumeric import cumsum as cumsum
from numpy.core.fromnumeric import diagonal as diagonal
from numpy.core.fromnumeric import mean as mean
from numpy.core.fromnumeric import ndim as ndim
from numpy.core.fromnumeric import nonzero as nonzero
from numpy.core.fromnumeric import partition as partition
from numpy.core.fromnumeric import prod as prod
from numpy.core.fromnumeric import ptp as ptp
from numpy.core.fromnumeric import put as put
from numpy.core.fromnumeric import ravel as ravel
from numpy.core.fromnumeric import repeat as repeat
from numpy.core.fromnumeric import reshape as reshape
from numpy.core.fromnumeric import resize as resize
from numpy.core.fromnumeric import searchsorted as searchsorted
from numpy.core.fromnumeric import shape as shape
from numpy.core.fromnumeric import size as size
from numpy.core.fromnumeric import sort as sort
from numpy.core.fromnumeric import squeeze as squeeze
from numpy.core.fromnumeric import std as std
from numpy.core.fromnumeric import sum as sum
from numpy.core.fromnumeric import swapaxes as swapaxes
from numpy.core.fromnumeric import take as take
from numpy.core.fromnumeric import trace as trace
from numpy.core.fromnumeric import transpose as transpose
from numpy.core.fromnumeric import var as var
from numpy.core.function_base import geomspace as geomspace
from numpy.core.function_base import linspace as linspace
from numpy.core.function_base import logspace as logspace
from numpy.core.numeric import allclose as allclose
from numpy.core.numeric import argwhere as argwhere
from numpy.core.numeric import array_equal as array_equal
from numpy.core.numeric import array_equiv as array_equiv
from numpy.core.numeric import base_repr as base_repr
from numpy.core.numeric import binary_repr as binary_repr
from numpy.core.numeric import convolve as convolve
from numpy.core.numeric import correlate as correlate
from numpy.core.numeric import count_nonzero as count_nonzero
from numpy.core.numeric import cross as cross
from numpy.core.numeric import empty_like as empty_like
from numpy.core.numeric import flatnonzero as flatnonzero
from numpy.core.numeric import fromfunction as fromfunction
from numpy.core.numeric import full as full
from numpy.core.numeric import full_like as full_like
from numpy.core.numeric import identity as identity
from numpy.core.numeric import indices as indices
from numpy.core.numeric import isclose as isclose
from numpy.core.numeric import isfortran as isfortran
from numpy.core.numeric import isscalar as isscalar
from numpy.core.numeric import moveaxis as moveaxis
from numpy.core.numeric import ones as ones
from numpy.core.numeric import ones_like as ones_like
from numpy.core.numeric import outer as outer
from numpy.core.numeric import roll as roll
from numpy.core.numeric import rollaxis as rollaxis
from numpy.core.numeric import tensordot as tensordot
from numpy.core.numeric import zeros_like as zeros_like
from numpy.core.numerictypes import ScalarType as ScalarType
from numpy.core.numerictypes import cast as cast
from numpy.core.numerictypes import find_common_type as find_common_type
from numpy.core.numerictypes import issctype as issctype
from numpy.core.numerictypes import issubclass_ as issubclass_
from numpy.core.numerictypes import issubdtype as issubdtype
from numpy.core.numerictypes import issubsctype as issubsctype
from numpy.core.numerictypes import maximum_sctype as maximum_sctype
from numpy.core.numerictypes import nbytes as nbytes
from numpy.core.numerictypes import obj2sctype as obj2sctype
from numpy.core.numerictypes import sctype2char as sctype2char
from numpy.core.numerictypes import typecodes as typecodes
from numpy.core.shape_base import atleast_1d as atleast_1d
from numpy.core.shape_base import atleast_2d as atleast_2d
from numpy.core.shape_base import atleast_3d as atleast_3d
from numpy.core.shape_base import block as block
from numpy.core.shape_base import hstack as hstack
from numpy.core.shape_base import stack as stack
from numpy.core.shape_base import vstack as vstack
from numpy.lib import emath as emath
from numpy.lib.arraypad import pad as pad
from numpy.lib.arraysetops import ediff1d as ediff1d
from numpy.lib.arraysetops import in1d as in1d
from numpy.lib.arraysetops import intersect1d as intersect1d
from numpy.lib.arraysetops import isin as isin
from numpy.lib.arraysetops import setdiff1d as setdiff1d
from numpy.lib.arraysetops import setxor1d as setxor1d
from numpy.lib.arraysetops import union1d as union1d
from numpy.lib.arraysetops import unique as unique
from numpy.lib.arrayterator import Arrayterator as Arrayterator
from numpy.lib.function_base import add_docstring as add_docstring
from numpy.lib.function_base import add_newdoc as add_newdoc
from numpy.lib.function_base import add_newdoc_ufunc as add_newdoc_ufunc
from numpy.lib.function_base import angle as angle
from numpy.lib.function_base import append as append
from numpy.lib.function_base import asarray_chkfinite as asarray_chkfinite
from numpy.lib.function_base import average as average
from numpy.lib.function_base import bartlett as bartlett
from numpy.lib.function_base import bincount as bincount
from numpy.lib.function_base import blackman as blackman
from numpy.lib.function_base import copy as copy
from numpy.lib.function_base import corrcoef as corrcoef
from numpy.lib.function_base import cov as cov
from numpy.lib.function_base import delete as delete
from numpy.lib.function_base import diff as diff
from numpy.lib.function_base import digitize as digitize
from numpy.lib.function_base import disp as disp
from numpy.lib.function_base import extract as extract
from numpy.lib.function_base import flip as flip
from numpy.lib.function_base import gradient as gradient
from numpy.lib.function_base import hamming as hamming
from numpy.lib.function_base import hanning as hanning
from numpy.lib.function_base import i0 as i0
from numpy.lib.function_base import insert as insert
from numpy.lib.function_base import interp as interp
from numpy.lib.function_base import iterable as iterable
from numpy.lib.function_base import kaiser as kaiser
from numpy.lib.function_base import median as median
from numpy.lib.function_base import meshgrid as meshgrid
from numpy.lib.function_base import msort as msort
from numpy.lib.function_base import percentile as percentile
from numpy.lib.function_base import piecewise as piecewise
from numpy.lib.function_base import place as place
from numpy.lib.function_base import quantile as quantile
from numpy.lib.function_base import rot90 as rot90
from numpy.lib.function_base import select as select
from numpy.lib.function_base import sinc as sinc
from numpy.lib.function_base import sort_complex as sort_complex
from numpy.lib.function_base import trapz as trapz
from numpy.lib.function_base import trim_zeros as trim_zeros
from numpy.lib.function_base import unwrap as unwrap
from numpy.lib.index_tricks import c_ as c_
from numpy.lib.index_tricks import diag_indices as diag_indices
from numpy.lib.index_tricks import diag_indices_from as diag_indices_from
from numpy.lib.index_tricks import fill_diagonal as fill_diagonal
from numpy.lib.index_tricks import index_exp as index_exp
from numpy.lib.index_tricks import ix_ as ix_
from numpy.lib.index_tricks import mgrid as mgrid
from numpy.lib.index_tricks import ogrid as ogrid
from numpy.lib.index_tricks import r_ as r_
from numpy.lib.index_tricks import ravel_multi_index as ravel_multi_index
from numpy.lib.index_tricks import s_ as s_
from numpy.lib.index_tricks import unravel_index as unravel_index
from numpy.lib.nanfunctions import nanargmax as nanargmax
from numpy.lib.nanfunctions import nanargmin as nanargmin
from numpy.lib.nanfunctions import nancumprod as nancumprod
from numpy.lib.nanfunctions import nancumsum as nancumsum
from numpy.lib.nanfunctions import nanmax as nanmax
from numpy.lib.nanfunctions import nanmean as nanmean
from numpy.lib.nanfunctions import nanmedian as nanmedian
from numpy.lib.nanfunctions import nanmin as nanmin
from numpy.lib.nanfunctions import nanpercentile as nanpercentile
from numpy.lib.nanfunctions import nanprod as nanprod
from numpy.lib.nanfunctions import nanquantile as nanquantile
from numpy.lib.nanfunctions import nanstd as nanstd
from numpy.lib.nanfunctions import nansum as nansum
from numpy.lib.nanfunctions import nanvar as nanvar
from numpy.lib.npyio import fromregex as fromregex
from numpy.lib.npyio import genfromtxt as genfromtxt
from numpy.lib.npyio import load as load
from numpy.lib.npyio import loads as loads
from numpy.lib.npyio import loadtxt as loadtxt
from numpy.lib.npyio import packbits as packbits
from numpy.lib.npyio import recfromcsv as recfromcsv
from numpy.lib.npyio import recfromtxt as recfromtxt
from numpy.lib.npyio import save as save
from numpy.lib.npyio import savetxt as savetxt
from numpy.lib.npyio import savez as savez
from numpy.lib.npyio import savez_compressed as savez_compressed
from numpy.lib.npyio import unpackbits as unpackbits
from numpy.lib.polynomial import poly as poly
from numpy.lib.polynomial import polyadd as polyadd
from numpy.lib.polynomial import polyder as polyder
from numpy.lib.polynomial import polydiv as polydiv
from numpy.lib.polynomial import polyfit as polyfit
from numpy.lib.polynomial import polyint as polyint
from numpy.lib.polynomial import polymul as polymul
from numpy.lib.polynomial import polysub as polysub
from numpy.lib.polynomial import polyval as polyval
from numpy.lib.polynomial import roots as roots
from numpy.lib.shape_base import apply_along_axis as apply_along_axis
from numpy.lib.shape_base import apply_over_axes as apply_over_axes
from numpy.lib.shape_base import array_split as array_split
from numpy.lib.shape_base import column_stack as column_stack
from numpy.lib.shape_base import dsplit as dsplit
from numpy.lib.shape_base import dstack as dstack
from numpy.lib.shape_base import expand_dims as expand_dims
from numpy.lib.shape_base import get_array_wrap as get_array_wrap
from numpy.lib.shape_base import hsplit as hsplit
from numpy.lib.shape_base import kron as kron
from numpy.lib.shape_base import put_along_axis as put_along_axis
from numpy.lib.shape_base import row_stack as row_stack
from numpy.lib.shape_base import split as split
from numpy.lib.shape_base import take_along_axis as take_along_axis
from numpy.lib.shape_base import tile as tile
from numpy.lib.shape_base import vsplit as vsplit
from numpy.lib.stride_tricks import broadcast_arrays as broadcast_arrays
from numpy.lib.stride_tricks import broadcast_shapes as broadcast_shapes
from numpy.lib.stride_tricks import broadcast_to as broadcast_to
from numpy.lib.twodim_base import diag as diag
from numpy.lib.twodim_base import diagflat as diagflat
from numpy.lib.twodim_base import eye as eye
from numpy.lib.twodim_base import fliplr as fliplr
from numpy.lib.twodim_base import flipud as flipud
from numpy.lib.twodim_base import histogram2d as histogram2d
from numpy.lib.twodim_base import mask_indices as mask_indices
from numpy.lib.twodim_base import tri as tri
from numpy.lib.twodim_base import tril as tril
from numpy.lib.twodim_base import tril_indices as tril_indices
from numpy.lib.twodim_base import tril_indices_from as tril_indices_from
from numpy.lib.twodim_base import triu as triu
from numpy.lib.twodim_base import triu_indices as triu_indices
from numpy.lib.twodim_base import triu_indices_from as triu_indices_from
from numpy.lib.twodim_base import vander as vander
from numpy.lib.type_check import asfarray as asfarray
from numpy.lib.type_check import common_type as common_type
from numpy.lib.type_check import imag as imag
from numpy.lib.type_check import iscomplex as iscomplex
from numpy.lib.type_check import iscomplexobj as iscomplexobj
from numpy.lib.type_check import isreal as isreal
from numpy.lib.type_check import isrealobj as isrealobj
from numpy.lib.type_check import mintypecode as mintypecode
from numpy.lib.type_check import nan_to_num as nan_to_num
from numpy.lib.type_check import real as real
from numpy.lib.type_check import real_if_close as real_if_close
from numpy.lib.type_check import typename as typename
from numpy.lib.ufunclike import fix as fix
from numpy.lib.ufunclike import isneginf as isneginf
from numpy.lib.ufunclike import isposinf as isposinf
from numpy.lib.utils import byte_bounds as byte_bounds
from numpy.lib.utils import deprecate as deprecate
from numpy.lib.utils import deprecate_with_doc as deprecate_with_doc
from numpy.lib.utils import get_include as get_include
from numpy.lib.utils import info as info
from numpy.lib.utils import issubclass_ as issubclass_
from numpy.lib.utils import issubdtype as issubdtype
from numpy.lib.utils import issubsctype as issubsctype
from numpy.lib.utils import lookfor as lookfor
from numpy.lib.utils import safe_eval as safe_eval
from numpy.lib.utils import source as source
from numpy.lib.utils import who as who
from numpy.typing import (
    ArrayLike,
    DTypeLike,
    NBitBase,
    NDArray,
    _8Bit,
    _16Bit,
    _32Bit,
    _64Bit,
    _80Bit,
    _96Bit,
    _128Bit,
    _256Bit,
    _ArrayLikeBool_co,
    _ArrayLikeComplex_co,
    _ArrayLikeDT64_co,
    _ArrayLikeFloat_co,
    _ArrayLikeInt_co,
    _ArrayLikeNumber_co,
    _ArrayLikeObject_co,
    _ArrayLikeTD64_co,
    _ArrayLikeUInt_co,
    _BoolCodes,
    _BoolLike_co,
    _ByteCodes,
    _BytesCodes,
    _CDoubleCodes,
    _CharLike_co,
    _CLongDoubleCodes,
    _Complex64Codes,
    _Complex128Codes,
    _ComplexLike_co,
    _CSingleCodes,
    _DoubleCodes,
    _DT64Codes,
    _Float16Codes,
    _Float32Codes,
    _Float64Codes,
    _FloatLike_co,
    _GUFunc_Nin2_Nout1,
    _HalfCodes,
    _Int8Codes,
    _Int16Codes,
    _Int32Codes,
    _Int64Codes,
    _IntCCodes,
    _IntCodes,
    _IntLike_co,
    _IntPCodes,
    _LongDoubleCodes,
    _LongLongCodes,
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
    _NestedSequence,
    _NumberLike_co,
    _ObjectCodes,
    _RecursiveSequence,
    _ScalarLike_co,
    _Shape,
    _ShapeLike,
    _ShortCodes,
    _SingleCodes,
    _StrCodes,
    _SupportsArray,
    _SupportsDType,
    _TD64Codes,
    _TD64Like_co,
    _UByteCodes,
    _UFunc_Nin1_Nout1,
    _UFunc_Nin1_Nout2,
    _UFunc_Nin2_Nout1,
    _UFunc_Nin2_Nout2,
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
    _VoidDTypeLike,
)
from numpy.typing._callable import (
    _BoolBitOp,
    _BoolDivMod,
    _BoolMod,
    _BoolOp,
    _BoolSub,
    _BoolTrueDiv,
    _ComparisonOp,
    _ComplexOp,
    _FloatDivMod,
    _FloatMod,
    _FloatOp,
    _IntTrueDiv,
    _NumberOp,
    _SignedIntBitOp,
    _SignedIntDivMod,
    _SignedIntMod,
    _SignedIntOp,
    _TD64Div,
    _UnsignedIntBitOp,
    _UnsignedIntDivMod,
    _UnsignedIntMod,
    _UnsignedIntOp,
)
from numpy.typing._extended_precision import complex160 as complex160
from numpy.typing._extended_precision import complex192 as complex192
from numpy.typing._extended_precision import complex256 as complex256
from numpy.typing._extended_precision import complex512 as complex512
from numpy.typing._extended_precision import float80 as float80
from numpy.typing._extended_precision import float96 as float96
from numpy.typing._extended_precision import float128 as float128
from numpy.typing._extended_precision import float256 as float256
from numpy.typing._extended_precision import int128 as int128
from numpy.typing._extended_precision import int256 as int256
from numpy.typing._extended_precision import uint128 as uint128
from numpy.typing._extended_precision import uint256 as uint256

if sys.version_info >= (3, 8): ...
else: ...
__all__: List[str]
__path__: List[str]
__version__: str
__git_version__: str

class MachAr:
    def __init__(
        self,
        float_conv: Any = ...,
        int_conv: Any = ...,
        float_to_float: Any = ...,
        float_to_str: Any = ...,
        title: Any = ...,
    ) -> None: ...
    def __getattr__(self, key: str) -> Any: ...

class busdaycalendar:
    def __new__(cls, weekmask: Any = ..., holidays: Any = ...) -> Any: ...
    def __getattr__(self, key: str) -> Any: ...

class chararray(ndarray[_ShapeType, _DType_co]):
    def __new__(
        subtype,
        shape: Any,
        itemsize: Any = ...,
        unicode: Any = ...,
        buffer: Any = ...,
        offset: Any = ...,
        strides: Any = ...,
        order: Any = ...,
    ) -> Any: ...
    def __array_finalize__(self, obj): ...
    def argsort(self, axis=..., kind=..., order=...): ...
    def capitalize(self): ...
    def center(self, width, fillchar=...): ...
    def count(self, sub, start=..., end=...): ...
    def decode(self, encoding=..., errors=...): ...
    def encode(self, encoding=..., errors=...): ...
    def endswith(self, suffix, start=..., end=...): ...
    def expandtabs(self, tabsize=...): ...
    def find(self, sub, start=..., end=...): ...
    def index(self, sub, start=..., end=...): ...
    def isalnum(self): ...
    def isalpha(self): ...
    def isdigit(self): ...
    def islower(self): ...
    def isspace(self): ...
    def istitle(self): ...
    def isupper(self): ...
    def join(self, seq): ...
    def ljust(self, width, fillchar=...): ...
    def lower(self): ...
    def lstrip(self, chars=...): ...
    def partition(self, sep): ...
    def replace(self, old, new, count=...): ...
    def rfind(self, sub, start=..., end=...): ...
    def rindex(self, sub, start=..., end=...): ...
    def rjust(self, width, fillchar=...): ...
    def rpartition(self, sep): ...
    def rsplit(self, sep=..., maxsplit=...): ...
    def rstrip(self, chars=...): ...
    def split(self, sep=..., maxsplit=...): ...
    def splitlines(self, keepends=...): ...
    def startswith(self, prefix, start=..., end=...): ...
    def strip(self, chars=...): ...
    def swapcase(self): ...
    def title(self): ...
    def translate(self, table, deletechars=...): ...
    def upper(self): ...
    def zfill(self, width): ...
    def isnumeric(self): ...
    def isdecimal(self): ...

class finfo:
    def __new__(cls, dtype: Any) -> Any: ...
    def __getattr__(self, key: str) -> Any: ...

class format_parser:
    def __init__(
        self,
        formats: Any,
        names: Any,
        titles: Any,
        aligned: Any = ...,
        byteorder: Any = ...,
    ) -> None: ...

class iinfo:
    def __init__(self, int_type: Any) -> None: ...
    def __getattr__(self, key: str) -> Any: ...

class matrix(ndarray[_ShapeType, _DType_co]):
    def __new__(subtype, data: Any, dtype: Any = ..., copy: Any = ...) -> Any: ...
    def __array_finalize__(self, obj): ...
    def __getitem__(self, index): ...
    def __mul__(self, other): ...
    def __rmul__(self, other): ...
    def __imul__(self, other): ...
    def __pow__(self, other): ...
    def __ipow__(self, other): ...
    def __rpow__(self, other): ...
    def tolist(self): ...
    def sum(self, axis=..., dtype=..., out=...): ...
    def squeeze(self, axis=...): ...
    def flatten(self, order=...): ...
    def mean(self, axis=..., dtype=..., out=...): ...
    def std(self, axis=..., dtype=..., out=..., ddof=...): ...
    def var(self, axis=..., dtype=..., out=..., ddof=...): ...
    def prod(self, axis=..., dtype=..., out=...): ...
    def any(self, axis=..., out=...): ...
    def all(self, axis=..., out=...): ...
    def max(self, axis=..., out=...): ...
    def argmax(self, axis=..., out=...): ...
    def min(self, axis=..., out=...): ...
    def argmin(self, axis=..., out=...): ...
    def ptp(self, axis=..., out=...): ...
    def ravel(self, order=...): ...
    @property
    def T(self): ...
    @property
    def I(self): ...
    @property
    def A(self): ...
    @property
    def A1(self): ...
    @property
    def H(self): ...
    def getT(self): ...
    def getA(self): ...
    def getA1(self): ...
    def getH(self): ...
    def getI(self): ...

class memmap(ndarray[_ShapeType, _DType_co]):
    def __new__(
        subtype,
        filename: Any,
        dtype: Any = ...,
        mode: Any = ...,
        offset: Any = ...,
        shape: Any = ...,
        order: Any = ...,
    ) -> Any: ...
    def __getattr__(self, key: str) -> Any: ...

class nditer:
    def __new__(
        cls,
        op: Any,
        flags: Any = ...,
        op_flags: Any = ...,
        op_dtypes: Any = ...,
        order: Any = ...,
        casting: Any = ...,
        op_axes: Any = ...,
        itershape: Any = ...,
        buffersize: Any = ...,
    ) -> Any: ...
    def __getattr__(self, key: str) -> Any: ...
    def __enter__(self) -> nditer: ...
    def __exit__(
        self,
        exc_type: None | Type[BaseException],
        exc_value: None | BaseException,
        traceback: None | TracebackType,
    ) -> None: ...
    def __iter__(self) -> Iterator[Any]: ...
    def __next__(self) -> Any: ...
    def __len__(self) -> int: ...
    def __copy__(self) -> nditer: ...
    def __getitem__(self, index: SupportsIndex | slice) -> Any: ...
    def __setitem__(self, index: SupportsIndex | slice, value: Any) -> None: ...
    def __delitem__(self, key: SupportsIndex | slice) -> None: ...

class poly1d:
    def __init__(self, c_or_r: Any, r: Any = ..., variable: Any = ...) -> None: ...
    def __call__(self, val: Any) -> Any: ...
    __hash__: Any
    @property
    def coeffs(self): ...
    @coeffs.setter
    def coeffs(self, value): ...
    @property
    def c(self): ...
    @c.setter
    def c(self, value): ...
    @property
    def coef(self): ...
    @coef.setter
    def coef(self, value): ...
    @property
    def coefficients(self): ...
    @coefficients.setter
    def coefficients(self, value): ...
    @property
    def variable(self): ...
    @property
    def order(self): ...
    @property
    def o(self): ...
    @property
    def roots(self): ...
    @property
    def r(self): ...
    def __array__(self, t=...): ...
    def __len__(self): ...
    def __neg__(self): ...
    def __pos__(self): ...
    def __mul__(self, other): ...
    def __rmul__(self, other): ...
    def __add__(self, other): ...
    def __radd__(self, other): ...
    def __pow__(self, val): ...
    def __sub__(self, other): ...
    def __rsub__(self, other): ...
    def __div__(self, other): ...
    def __truediv__(self, other): ...
    def __rdiv__(self, other): ...
    def __rtruediv__(self, other): ...
    def __eq__(self, other) -> bool: ...
    def __ne__(self, other) -> bool: ...
    def __getitem__(self, val): ...
    def __setitem__(self, key, val): ...
    def __iter__(self): ...
    def integ(self, m=..., k=...): ...
    def deriv(self, m=...): ...

class recarray(ndarray[_ShapeType, _DType_co]):
    def __new__(
        subtype,
        shape: Any,
        dtype: Any = ...,
        buf: Any = ...,
        offset: Any = ...,
        strides: Any = ...,
        formats: Any = ...,
        names: Any = ...,
        titles: Any = ...,
        byteorder: Any = ...,
        aligned: Any = ...,
        order: Any = ...,
    ) -> Any: ...
    def __array_finalize__(self, obj): ...
    def __getattribute__(self, attr): ...
    def __setattr__(self, attr, val): ...
    def __getitem__(self, indx): ...
    def field(self, attr, val=...): ...

class record(void):
    def __getattribute__(self, attr): ...
    def __setattr__(self, attr, val): ...
    def __getitem__(self, indx): ...
    def pprint(self): ...

class vectorize:
    pyfunc: Any
    cache: Any
    signature: Any
    otypes: Any
    excluded: Any
    __doc__: Any
    def __init__(
        self,
        pyfunc,
        otypes: Any = ...,
        doc: Any = ...,
        excluded: Any = ...,
        cache: Any = ...,
        signature: Any = ...,
    ) -> None: ...
    def __call__(self, *args: Any, **kwargs: Any) -> Any: ...

def asmatrix(data, dtype=...): ...
def asscalar(a): ...
def cumproduct(*args, **kwargs): ...
def histogram(a, bins=..., range=..., normed=..., weights=..., density=...): ...
def histogram_bin_edges(a, bins=..., range=..., weights=...): ...
def histogramdd(sample, bins=..., range=..., normed=..., weights=..., density=...): ...
def mat(data, dtype=...): ...
def max(a, axis=..., out=..., keepdims=..., initial=..., where=...): ...
def min(a, axis=..., out=..., keepdims=..., initial=..., where=...): ...
def product(*args, **kwargs): ...
def round(a, decimals=..., out=...): ...
def round_(a, decimals=..., out=...): ...
def show_config(): ...
@overload
def arange(stop, dtype=..., *, like=...): ...
@overload
def arange(start, stop, step=..., dtype=..., *, like=...): ...
def busday_count(
    begindates, enddates, weekmask=..., holidays=..., busdaycal=..., out=...
): ...
def busday_offset(
    dates, offsets, roll=..., weekmask=..., holidays=..., busdaycal=..., out=...
): ...
def can_cast(from_, to, casting=...): ...
def compare_chararrays(a, b, cmp_op, rstrip): ...
def concatenate(__a, axis=..., out=..., dtype=..., casting=...): ...
def copyto(dst, src, casting=..., where=...): ...
def datetime_as_string(arr, unit=..., timezone=..., casting=...): ...
def datetime_data(__dtype): ...
def dot(a, b, out=...): ...
def frombuffer(buffer, dtype=..., count=..., offset=..., *, like=...): ...
def fromfile(file, dtype=..., count=..., sep=..., offset=..., *, like=...): ...
def fromiter(iter, dtype, count=..., *, like=...): ...
def frompyfunc(func, nin, nout, *identity): ...
def fromstring(string, dtype=..., count=..., sep=..., *, like=...): ...
def geterrobj(): ...
def inner(a, b): ...
def is_busday(dates, weekmask=..., holidays=..., busdaycal=..., out=...): ...
def lexsort(keys, axis=...): ...
def may_share_memory(a, b, max_work=...): ...
def min_scalar_type(a): ...
def nested_iters(*args, **kwargs): ...
def promote_types(type1, type2): ...
def putmask(a, mask, values): ...
def result_type(*arrays_and_dtypes): ...
def seterrobj(errobj): ...
def shares_memory(a, b, max_work=...): ...
def vdot(a, b): ...
@overload
def where(__condition): ...
@overload
def where(__condition, __x, __y): ...

_NdArraySubClass = ...
_DTypeScalar_co = ...
_ByteOrder = L["S", "<", ">", "=", "|", "L", "B", "N", "I"]

class dtype(Generic[_DTypeScalar_co]):
    names: Optional[Tuple[builtins.str, ...]]
    @overload
    def __new__(
        cls, dtype: Type[_DTypeScalar_co], align: bool = ..., copy: bool = ...
    ) -> dtype[_DTypeScalar_co]: ...
    @overload
    def __new__(
        cls, dtype: Type[bool], align: bool = ..., copy: bool = ...
    ) -> dtype[bool_]: ...
    @overload
    def __new__(
        cls, dtype: Type[int], align: bool = ..., copy: bool = ...
    ) -> dtype[int_]: ...
    @overload
    def __new__(
        cls, dtype: Optional[Type[float]], align: bool = ..., copy: bool = ...
    ) -> dtype[float_]: ...
    @overload
    def __new__(
        cls, dtype: Type[complex], align: bool = ..., copy: bool = ...
    ) -> dtype[complex_]: ...
    @overload
    def __new__(
        cls, dtype: Type[builtins.str], align: bool = ..., copy: bool = ...
    ) -> dtype[str_]: ...
    @overload
    def __new__(
        cls, dtype: Type[bytes], align: bool = ..., copy: bool = ...
    ) -> dtype[bytes_]: ...
    @overload
    def __new__(
        cls, dtype: _UInt8Codes | Type[ct.c_uint8], align: bool = ..., copy: bool = ...
    ) -> dtype[uint8]: ...
    @overload
    def __new__(
        cls, dtype: _UInt16Codes | Type[ct.c_uint16], align: bool = ..., copy: bool = ...
    ) -> dtype[uint16]: ...
    @overload
    def __new__(
        cls, dtype: _UInt32Codes | Type[ct.c_uint32], align: bool = ..., copy: bool = ...
    ) -> dtype[uint32]: ...
    @overload
    def __new__(
        cls, dtype: _UInt64Codes | Type[ct.c_uint64], align: bool = ..., copy: bool = ...
    ) -> dtype[uint64]: ...
    @overload
    def __new__(
        cls, dtype: _UByteCodes | Type[ct.c_ubyte], align: bool = ..., copy: bool = ...
    ) -> dtype[ubyte]: ...
    @overload
    def __new__(
        cls, dtype: _UShortCodes | Type[ct.c_ushort], align: bool = ..., copy: bool = ...
    ) -> dtype[ushort]: ...
    @overload
    def __new__(
        cls, dtype: _UIntCCodes | Type[ct.c_uint], align: bool = ..., copy: bool = ...
    ) -> dtype[uintc]: ...
    @overload
    def __new__(
        cls,
        dtype: _UIntPCodes | Type[ct.c_void_p] | Type[ct.c_size_t],
        align: bool = ...,
        copy: bool = ...,
    ) -> dtype[uintp]: ...
    @overload
    def __new__(
        cls, dtype: _UIntCodes | Type[ct.c_ulong], align: bool = ..., copy: bool = ...
    ) -> dtype[uint]: ...
    @overload
    def __new__(
        cls,
        dtype: _ULongLongCodes | Type[ct.c_ulonglong],
        align: bool = ...,
        copy: bool = ...,
    ) -> dtype[ulonglong]: ...
    @overload
    def __new__(
        cls, dtype: _Int8Codes | Type[ct.c_int8], align: bool = ..., copy: bool = ...
    ) -> dtype[int8]: ...
    @overload
    def __new__(
        cls, dtype: _Int16Codes | Type[ct.c_int16], align: bool = ..., copy: bool = ...
    ) -> dtype[int16]: ...
    @overload
    def __new__(
        cls, dtype: _Int32Codes | Type[ct.c_int32], align: bool = ..., copy: bool = ...
    ) -> dtype[int32]: ...
    @overload
    def __new__(
        cls, dtype: _Int64Codes | Type[ct.c_int64], align: bool = ..., copy: bool = ...
    ) -> dtype[int64]: ...
    @overload
    def __new__(
        cls, dtype: _ByteCodes | Type[ct.c_byte], align: bool = ..., copy: bool = ...
    ) -> dtype[byte]: ...
    @overload
    def __new__(
        cls, dtype: _ShortCodes | Type[ct.c_short], align: bool = ..., copy: bool = ...
    ) -> dtype[short]: ...
    @overload
    def __new__(
        cls, dtype: _IntCCodes | Type[ct.c_int], align: bool = ..., copy: bool = ...
    ) -> dtype[intc]: ...
    @overload
    def __new__(
        cls, dtype: _IntPCodes | Type[ct.c_ssize_t], align: bool = ..., copy: bool = ...
    ) -> dtype[intp]: ...
    @overload
    def __new__(
        cls, dtype: _IntCodes | Type[ct.c_long], align: bool = ..., copy: bool = ...
    ) -> dtype[int_]: ...
    @overload
    def __new__(
        cls,
        dtype: _LongLongCodes | Type[ct.c_longlong],
        align: bool = ...,
        copy: bool = ...,
    ) -> dtype[longlong]: ...
    @overload
    def __new__(
        cls, dtype: _Float16Codes, align: bool = ..., copy: bool = ...
    ) -> dtype[float16]: ...
    @overload
    def __new__(
        cls, dtype: _Float32Codes, align: bool = ..., copy: bool = ...
    ) -> dtype[float32]: ...
    @overload
    def __new__(
        cls, dtype: _Float64Codes, align: bool = ..., copy: bool = ...
    ) -> dtype[float64]: ...
    @overload
    def __new__(
        cls, dtype: _HalfCodes, align: bool = ..., copy: bool = ...
    ) -> dtype[half]: ...
    @overload
    def __new__(
        cls, dtype: _SingleCodes | Type[ct.c_float], align: bool = ..., copy: bool = ...
    ) -> dtype[single]: ...
    @overload
    def __new__(
        cls, dtype: _DoubleCodes | Type[ct.c_double], align: bool = ..., copy: bool = ...
    ) -> dtype[double]: ...
    @overload
    def __new__(
        cls,
        dtype: _LongDoubleCodes | Type[ct.c_longdouble],
        align: bool = ...,
        copy: bool = ...,
    ) -> dtype[longdouble]: ...
    @overload
    def __new__(
        cls, dtype: _Complex64Codes, align: bool = ..., copy: bool = ...
    ) -> dtype[complex64]: ...
    @overload
    def __new__(
        cls, dtype: _Complex128Codes, align: bool = ..., copy: bool = ...
    ) -> dtype[complex128]: ...
    @overload
    def __new__(
        cls, dtype: _CSingleCodes, align: bool = ..., copy: bool = ...
    ) -> dtype[csingle]: ...
    @overload
    def __new__(
        cls, dtype: _CDoubleCodes, align: bool = ..., copy: bool = ...
    ) -> dtype[cdouble]: ...
    @overload
    def __new__(
        cls, dtype: _CLongDoubleCodes, align: bool = ..., copy: bool = ...
    ) -> dtype[clongdouble]: ...
    @overload
    def __new__(
        cls, dtype: _BoolCodes | Type[ct.c_bool], align: bool = ..., copy: bool = ...
    ) -> dtype[bool_]: ...
    @overload
    def __new__(
        cls, dtype: _TD64Codes, align: bool = ..., copy: bool = ...
    ) -> dtype[timedelta64]: ...
    @overload
    def __new__(
        cls, dtype: _DT64Codes, align: bool = ..., copy: bool = ...
    ) -> dtype[datetime64]: ...
    @overload
    def __new__(
        cls, dtype: _StrCodes, align: bool = ..., copy: bool = ...
    ) -> dtype[str_]: ...
    @overload
    def __new__(
        cls, dtype: _BytesCodes | Type[ct.c_char], align: bool = ..., copy: bool = ...
    ) -> dtype[bytes_]: ...
    @overload
    def __new__(
        cls, dtype: _VoidCodes, align: bool = ..., copy: bool = ...
    ) -> dtype[void]: ...
    @overload
    def __new__(
        cls, dtype: _ObjectCodes | Type[ct.py_object], align: bool = ..., copy: bool = ...
    ) -> dtype[object_]: ...
    @overload
    def __new__(
        cls, dtype: dtype[_DTypeScalar_co], align: bool = ..., copy: bool = ...
    ) -> dtype[_DTypeScalar_co]: ...
    @overload
    def __new__(
        cls,
        dtype: _SupportsDType[dtype[_DTypeScalar_co]],
        align: bool = ...,
        copy: bool = ...,
    ) -> dtype[_DTypeScalar_co]: ...
    @overload
    def __new__(
        cls, dtype: builtins.str, align: bool = ..., copy: bool = ...
    ) -> dtype[Any]: ...
    @overload
    def __new__(
        cls, dtype: _VoidDTypeLike, align: bool = ..., copy: bool = ...
    ) -> dtype[void]: ...
    @overload
    def __new__(
        cls, dtype: Type[object], align: bool = ..., copy: bool = ...
    ) -> dtype[object_]: ...
    @overload
    def __getitem__(self: dtype[void], key: List[builtins.str]) -> dtype[void]: ...
    @overload
    def __getitem__(self: dtype[void], key: Union[builtins.str, int]) -> dtype[Any]: ...
    @overload
    def __mul__(self, value: L[0]) -> None: ...
    @overload
    def __mul__(self: _DType, value: L[1]) -> _DType: ...
    @overload
    def __mul__(self, value: int) -> dtype[void]: ...
    def __rmul__(self, value: int) -> Any: ...
    def __gt__(self, other: DTypeLike) -> bool: ...
    def __ge__(self, other: DTypeLike) -> bool: ...
    def __lt__(self, other: DTypeLike) -> bool: ...
    def __le__(self, other: DTypeLike) -> bool: ...
    @property
    def alignment(self) -> int: ...
    @property
    def base(self: _DType) -> _DType: ...
    @property
    def byteorder(self) -> builtins.str: ...
    @property
    def char(self) -> builtins.str: ...
    @property
    def descr(
        self,
    ) -> List[
        Union[
            Tuple[builtins.str, builtins.str], Tuple[builtins.str, builtins.str, _Shape]
        ]
    ]: ...
    @property
    def fields(
        self,
    ) -> Optional[
        Mapping[builtins.str, Union[Tuple[dtype[Any], int], Tuple[dtype[Any], int, Any]]]
    ]: ...
    @property
    def flags(self) -> int: ...
    @property
    def hasobject(self) -> bool: ...
    @property
    def isbuiltin(self) -> int: ...
    @property
    def isnative(self) -> bool: ...
    @property
    def isalignedstruct(self) -> bool: ...
    @property
    def itemsize(self) -> int: ...
    @property
    def kind(self) -> builtins.str: ...
    @property
    def metadata(self) -> Optional[Mapping[builtins.str, Any]]: ...
    @property
    def name(self) -> builtins.str: ...
    @property
    def names(self) -> Optional[Tuple[str, ...]]: ...
    @property
    def num(self) -> int: ...
    @property
    def shape(self) -> _Shape: ...
    @property
    def ndim(self) -> int: ...
    @property
    def subdtype(self: _DType) -> Optional[Tuple[_DType, _Shape]]: ...
    def newbyteorder(self: _DType, __new_order: _ByteOrder = ...) -> _DType: ...
    @property
    def str(self) -> builtins.str: ...
    @property
    def type(self) -> Type[_DTypeScalar_co]: ...

class _flagsobj:
    aligned: bool
    updateifcopy: bool
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
    def __getitem__(self, key: str) -> bool: ...
    def __setitem__(self, key: str, value: bool) -> None: ...

_ArrayLikeInt = Union[
    int, integer, Sequence[Union[int, integer]], Sequence[Sequence[Any]], ndarray
]
_FlatIterSelf = ...

class flatiter(Generic[_NdArraySubClass]):
    @property
    def base(self) -> _NdArraySubClass: ...
    @property
    def coords(self) -> _Shape: ...
    @property
    def index(self) -> int: ...
    def copy(self) -> _NdArraySubClass: ...
    def __iter__(self: _FlatIterSelf) -> _FlatIterSelf: ...
    def __next__(self: flatiter[ndarray[Any, dtype[_ScalarType]]]) -> _ScalarType: ...
    def __len__(self) -> int: ...
    @overload
    def __getitem__(
        self: flatiter[ndarray[Any, dtype[_ScalarType]]], key: Union[int, integer]
    ) -> _ScalarType: ...
    @overload
    def __getitem__(
        self, key: Union[_ArrayLikeInt, slice, ellipsis]
    ) -> _NdArraySubClass: ...
    @overload
    def __array__(
        self: flatiter[ndarray[Any, _DType]], __dtype: None = ...
    ) -> ndarray[Any, _DType]: ...
    @overload
    def __array__(self, __dtype: _DType) -> ndarray[Any, _DType]: ...

_OrderKACF = Optional[L["K", "A", "C", "F"]]
_OrderACF = Optional[L["A", "C", "F"]]
_OrderCF = Optional[L["C", "F"]]
_ModeKind = L["raise", "wrap", "clip"]
_PartitionKind = L["introselect"]
_SortKind = L["quicksort", "mergesort", "heapsort", "stable"]
_SortSide = L["left", "right"]
_ArraySelf = ...

class _ArrayOrScalarCommon:
    @property
    def T(self: _ArraySelf) -> _ArraySelf: ...
    @property
    def data(self) -> memoryview: ...
    @property
    def flags(self) -> _flagsobj: ...
    @property
    def itemsize(self) -> int: ...
    @property
    def nbytes(self) -> int: ...
    def __bool__(self) -> bool: ...
    def __bytes__(self) -> bytes: ...
    def __str__(self) -> str: ...
    def __repr__(self) -> str: ...
    def __copy__(self: _ArraySelf) -> _ArraySelf: ...
    def __deepcopy__(self: _ArraySelf, __memo: Optional[dict] = ...) -> _ArraySelf: ...
    def __eq__(self, other) -> bool: ...
    def __ne__(self, other) -> bool: ...
    def copy(self: _ArraySelf, order: _OrderKACF = ...) -> _ArraySelf: ...
    def dump(self, file: str) -> None: ...
    def dumps(self) -> bytes: ...
    def tobytes(self, order: _OrderKACF = ...) -> bytes: ...
    def tofile(
        self,
        fid: Union[IO[bytes], str, bytes, os.PathLike[Any]],
        sep: str = ...,
        format: str = ...,
    ) -> None: ...
    def tolist(self) -> Any: ...
    def __getitem__(self, key) -> Any: ...
    @property
    def __array_interface__(self): ...
    @property
    def __array_priority__(self): ...
    @property
    def __array_struct__(self): ...
    def __array_wrap__(array, context=...): ...
    def __setstate__(self, __state): ...
    @overload
    def all(
        self, axis: None = ..., out: None = ..., keepdims: L[False] = ...
    ) -> bool_: ...
    @overload
    def all(
        self, axis: Optional[_ShapeLike] = ..., out: None = ..., keepdims: bool = ...
    ) -> Any: ...
    @overload
    def all(
        self,
        axis: Optional[_ShapeLike] = ...,
        out: _NdArraySubClass = ...,
        keepdims: bool = ...,
    ) -> _NdArraySubClass: ...
    @overload
    def any(
        self, axis: None = ..., out: None = ..., keepdims: L[False] = ...
    ) -> bool_: ...
    @overload
    def any(
        self, axis: Optional[_ShapeLike] = ..., out: None = ..., keepdims: bool = ...
    ) -> Any: ...
    @overload
    def any(
        self,
        axis: Optional[_ShapeLike] = ...,
        out: _NdArraySubClass = ...,
        keepdims: bool = ...,
    ) -> _NdArraySubClass: ...
    @overload
    def argmax(self, axis: None = ..., out: None = ...) -> intp: ...
    @overload
    def argmax(self, axis: _ShapeLike = ..., out: None = ...) -> Any: ...
    @overload
    def argmax(
        self, axis: Optional[_ShapeLike] = ..., out: _NdArraySubClass = ...
    ) -> _NdArraySubClass: ...
    @overload
    def argmin(self, axis: None = ..., out: None = ...) -> intp: ...
    @overload
    def argmin(self, axis: _ShapeLike = ..., out: None = ...) -> Any: ...
    @overload
    def argmin(
        self, axis: Optional[_ShapeLike] = ..., out: _NdArraySubClass = ...
    ) -> _NdArraySubClass: ...
    def argsort(
        self,
        axis: Optional[SupportsIndex] = ...,
        kind: Optional[_SortKind] = ...,
        order: Union[None, str, Sequence[str]] = ...,
    ) -> ndarray: ...
    @overload
    def choose(
        self, choices: ArrayLike, out: None = ..., mode: _ModeKind = ...
    ) -> ndarray: ...
    @overload
    def choose(
        self, choices: ArrayLike, out: _NdArraySubClass = ..., mode: _ModeKind = ...
    ) -> _NdArraySubClass: ...
    @overload
    def clip(
        self,
        min: ArrayLike = ...,
        max: Optional[ArrayLike] = ...,
        out: None = ...,
        **kwargs: Any
    ) -> ndarray: ...
    @overload
    def clip(
        self, min: None = ..., max: ArrayLike = ..., out: None = ..., **kwargs: Any
    ) -> ndarray: ...
    @overload
    def clip(
        self,
        min: ArrayLike = ...,
        max: Optional[ArrayLike] = ...,
        out: _NdArraySubClass = ...,
        **kwargs: Any
    ) -> _NdArraySubClass: ...
    @overload
    def clip(
        self,
        min: None = ...,
        max: ArrayLike = ...,
        out: _NdArraySubClass = ...,
        **kwargs: Any
    ) -> _NdArraySubClass: ...
    @overload
    def compress(
        self, a: ArrayLike, axis: Optional[SupportsIndex] = ..., out: None = ...
    ) -> ndarray: ...
    @overload
    def compress(
        self,
        a: ArrayLike,
        axis: Optional[SupportsIndex] = ...,
        out: _NdArraySubClass = ...,
    ) -> _NdArraySubClass: ...
    def conj(self: _ArraySelf) -> _ArraySelf: ...
    def conjugate(self: _ArraySelf) -> _ArraySelf: ...
    @overload
    def cumprod(
        self, axis: Optional[SupportsIndex] = ..., dtype: DTypeLike = ..., out: None = ...
    ) -> ndarray: ...
    @overload
    def cumprod(
        self,
        axis: Optional[SupportsIndex] = ...,
        dtype: DTypeLike = ...,
        out: _NdArraySubClass = ...,
    ) -> _NdArraySubClass: ...
    @overload
    def cumsum(
        self, axis: Optional[SupportsIndex] = ..., dtype: DTypeLike = ..., out: None = ...
    ) -> ndarray: ...
    @overload
    def cumsum(
        self,
        axis: Optional[SupportsIndex] = ...,
        dtype: DTypeLike = ...,
        out: _NdArraySubClass = ...,
    ) -> _NdArraySubClass: ...
    @overload
    def max(
        self,
        axis: Optional[_ShapeLike] = ...,
        out: None = ...,
        keepdims: bool = ...,
        initial: _NumberLike_co = ...,
        where: _ArrayLikeBool_co = ...,
    ) -> Any: ...
    @overload
    def max(
        self,
        axis: Optional[_ShapeLike] = ...,
        out: _NdArraySubClass = ...,
        keepdims: bool = ...,
        initial: _NumberLike_co = ...,
        where: _ArrayLikeBool_co = ...,
    ) -> _NdArraySubClass: ...
    @overload
    def mean(
        self,
        axis: Optional[_ShapeLike] = ...,
        dtype: DTypeLike = ...,
        out: None = ...,
        keepdims: bool = ...,
    ) -> Any: ...
    @overload
    def mean(
        self,
        axis: Optional[_ShapeLike] = ...,
        dtype: DTypeLike = ...,
        out: _NdArraySubClass = ...,
        keepdims: bool = ...,
    ) -> _NdArraySubClass: ...
    @overload
    def min(
        self,
        axis: Optional[_ShapeLike] = ...,
        out: None = ...,
        keepdims: bool = ...,
        initial: _NumberLike_co = ...,
        where: _ArrayLikeBool_co = ...,
    ) -> Any: ...
    @overload
    def min(
        self,
        axis: Optional[_ShapeLike] = ...,
        out: _NdArraySubClass = ...,
        keepdims: bool = ...,
        initial: _NumberLike_co = ...,
        where: _ArrayLikeBool_co = ...,
    ) -> _NdArraySubClass: ...
    def newbyteorder(self: _ArraySelf, __new_order: _ByteOrder = ...) -> _ArraySelf: ...
    @overload
    def prod(
        self,
        axis: Optional[_ShapeLike] = ...,
        dtype: DTypeLike = ...,
        out: None = ...,
        keepdims: bool = ...,
        initial: _NumberLike_co = ...,
        where: _ArrayLikeBool_co = ...,
    ) -> Any: ...
    @overload
    def prod(
        self,
        axis: Optional[_ShapeLike] = ...,
        dtype: DTypeLike = ...,
        out: _NdArraySubClass = ...,
        keepdims: bool = ...,
        initial: _NumberLike_co = ...,
        where: _ArrayLikeBool_co = ...,
    ) -> _NdArraySubClass: ...
    @overload
    def ptp(
        self, axis: Optional[_ShapeLike] = ..., out: None = ..., keepdims: bool = ...
    ) -> Any: ...
    @overload
    def ptp(
        self,
        axis: Optional[_ShapeLike] = ...,
        out: _NdArraySubClass = ...,
        keepdims: bool = ...,
    ) -> _NdArraySubClass: ...
    @overload
    def round(
        self: _ArraySelf, decimals: SupportsIndex = ..., out: None = ...
    ) -> _ArraySelf: ...
    @overload
    def round(
        self, decimals: SupportsIndex = ..., out: _NdArraySubClass = ...
    ) -> _NdArraySubClass: ...
    @overload
    def std(
        self,
        axis: Optional[_ShapeLike] = ...,
        dtype: DTypeLike = ...,
        out: None = ...,
        ddof: int = ...,
        keepdims: bool = ...,
    ) -> Any: ...
    @overload
    def std(
        self,
        axis: Optional[_ShapeLike] = ...,
        dtype: DTypeLike = ...,
        out: _NdArraySubClass = ...,
        ddof: int = ...,
        keepdims: bool = ...,
    ) -> _NdArraySubClass: ...
    @overload
    def sum(
        self,
        axis: Optional[_ShapeLike] = ...,
        dtype: DTypeLike = ...,
        out: None = ...,
        keepdims: bool = ...,
        initial: _NumberLike_co = ...,
        where: _ArrayLikeBool_co = ...,
    ) -> Any: ...
    @overload
    def sum(
        self,
        axis: Optional[_ShapeLike] = ...,
        dtype: DTypeLike = ...,
        out: _NdArraySubClass = ...,
        keepdims: bool = ...,
        initial: _NumberLike_co = ...,
        where: _ArrayLikeBool_co = ...,
    ) -> _NdArraySubClass: ...
    @overload
    def var(
        self,
        axis: Optional[_ShapeLike] = ...,
        dtype: DTypeLike = ...,
        out: None = ...,
        ddof: int = ...,
        keepdims: bool = ...,
    ) -> Any: ...
    @overload
    def var(
        self,
        axis: Optional[_ShapeLike] = ...,
        dtype: DTypeLike = ...,
        out: _NdArraySubClass = ...,
        ddof: int = ...,
        keepdims: bool = ...,
    ) -> _NdArraySubClass: ...

_DType = ...
_DType_co = ...
_ShapeType = ...
_NumberType = ...
_BufferType = Union[ndarray, bytes, bytearray, memoryview]
_T = ...
_T_co = ...
_2Tuple = Tuple[_T, _T]
_Casting = L["no", "equiv", "safe", "same_kind", "unsafe"]
_DTypeLike = (
    Union[dtype[_ScalarType], Type[_ScalarType], _SupportsDType[dtype[_ScalarType]]],
)
_ArrayUInt_co = NDArray[Union[bool_, unsignedinteger[Any]]]
_ArrayInt_co = NDArray[Union[bool_, integer[Any]]]
_ArrayFloat_co = NDArray[Union[bool_, integer[Any], floating[Any]]]
_ArrayComplex_co = NDArray[
    Union[bool_, integer[Any], floating[Any], complexfloating[Any, Any]]
]
_ArrayNumber_co = NDArray[Union[bool_, number[Any]]]
_ArrayTD64_co = NDArray[Union[bool_, integer[Any], timedelta64]]
_dtype = dtype

class _SupportsItem(Protocol[_T_co]):
    def item(self, __args: Any) -> _T_co: ...

class _SupportsReal(Protocol[_T_co]):
    @property
    def real(self) -> _T_co: ...

class _SupportsImag(Protocol[_T_co]):
    @property
    def imag(self) -> _T_co: ...

class ndarray(_ArrayOrScalarCommon, Generic[_ShapeType, _DType_co]):
    @property
    def base(self) -> Optional[ndarray]: ...
    @property
    def ndim(self) -> int: ...
    @property
    def size(self) -> int: ...
    @property
    def real(
        self: NDArray[_SupportsReal[_ScalarType]],
    ) -> ndarray[_ShapeType, _dtype[_ScalarType]]: ...
    @real.setter
    def real(self, value: ArrayLike) -> None: ...
    @property
    def imag(
        self: NDArray[_SupportsImag[_ScalarType]],
    ) -> ndarray[_ShapeType, _dtype[_ScalarType]]: ...
    @imag.setter
    def imag(self, value: ArrayLike) -> None: ...
    def __new__(
        cls: Type[_ArraySelf],
        shape: _ShapeLike,
        dtype: DTypeLike = ...,
        buffer: _BufferType = ...,
        offset: int = ...,
        strides: _ShapeLike = ...,
        order: _OrderKACF = ...,
    ) -> _ArraySelf: ...
    @overload
    def __array__(self, __dtype: None = ...) -> ndarray[Any, _DType_co]: ...
    @overload
    def __array__(self, __dtype: _DType) -> ndarray[Any, _DType]: ...
    @property
    def ctypes(self) -> _ctypes[int]: ...
    @property
    def shape(self) -> _Shape: ...
    @shape.setter
    def shape(self, value: _ShapeLike) -> None: ...
    @property
    def strides(self) -> _Shape: ...
    @strides.setter
    def strides(self, value: _ShapeLike) -> None: ...
    def byteswap(self: _ArraySelf, inplace: bool = ...) -> _ArraySelf: ...
    def fill(self, value: Any) -> None: ...
    @property
    def flat(self: _NdArraySubClass) -> flatiter[_NdArraySubClass]: ...
    @overload
    def item(
        self: ndarray[Any, _dtype[_SupportsItem[_T]]], *args: SupportsIndex
    ) -> _T: ...
    @overload
    def item(
        self: ndarray[Any, _dtype[_SupportsItem[_T]]], __args: Tuple[SupportsIndex, ...]
    ) -> _T: ...
    @overload
    def itemset(self, __value: Any) -> None: ...
    @overload
    def itemset(self, __item: _ShapeLike, __value: Any) -> None: ...
    @overload
    def resize(self, __new_shape: _ShapeLike, *, refcheck: bool = ...) -> None: ...
    @overload
    def resize(self, *new_shape: SupportsIndex, refcheck: bool = ...) -> None: ...
    def setflags(self, write: bool = ..., align: bool = ..., uic: bool = ...) -> None: ...
    def squeeze(
        self, axis: Union[SupportsIndex, Tuple[SupportsIndex, ...]] = ...
    ) -> ndarray[Any, _DType_co]: ...
    def swapaxes(
        self, axis1: SupportsIndex, axis2: SupportsIndex
    ) -> ndarray[Any, _DType_co]: ...
    @overload
    def transpose(self: _ArraySelf, __axes: _ShapeLike) -> _ArraySelf: ...
    @overload
    def transpose(self: _ArraySelf, *axes: SupportsIndex) -> _ArraySelf: ...
    def argpartition(
        self,
        kth: _ArrayLikeInt_co,
        axis: Optional[SupportsIndex] = ...,
        kind: _PartitionKind = ...,
        order: Union[None, str, Sequence[str]] = ...,
    ) -> ndarray[Any, _dtype[intp]]: ...
    def diagonal(
        self,
        offset: SupportsIndex = ...,
        axis1: SupportsIndex = ...,
        axis2: SupportsIndex = ...,
    ) -> ndarray[Any, _DType_co]: ...
    @overload
    def dot(self, b: _ScalarLike_co, out: None = ...) -> ndarray: ...
    @overload
    def dot(self, b: ArrayLike, out: None = ...) -> Any: ...
    @overload
    def dot(self, b: ArrayLike, out: _NdArraySubClass) -> _NdArraySubClass: ...
    def nonzero(self) -> Tuple[ndarray[Any, _dtype[intp]], ...]: ...
    def partition(
        self,
        kth: _ArrayLikeInt_co,
        axis: SupportsIndex = ...,
        kind: _PartitionKind = ...,
        order: Union[None, str, Sequence[str]] = ...,
    ) -> None: ...
    def put(self, ind: _ArrayLikeInt_co, v: ArrayLike, mode: _ModeKind = ...) -> None: ...
    @overload
    def searchsorted(
        self,
        v: _ScalarLike_co,
        side: _SortSide = ...,
        sorter: Optional[_ArrayLikeInt_co] = ...,
    ) -> intp: ...
    @overload
    def searchsorted(
        self,
        v: ArrayLike,
        side: _SortSide = ...,
        sorter: Optional[_ArrayLikeInt_co] = ...,
    ) -> ndarray[Any, _dtype[intp]]: ...
    def setfield(
        self, val: ArrayLike, dtype: DTypeLike, offset: SupportsIndex = ...
    ) -> None: ...
    def sort(
        self,
        axis: SupportsIndex = ...,
        kind: Optional[_SortKind] = ...,
        order: Union[None, str, Sequence[str]] = ...,
    ) -> None: ...
    @overload
    def trace(
        self,
        offset: SupportsIndex = ...,
        axis1: SupportsIndex = ...,
        axis2: SupportsIndex = ...,
        dtype: DTypeLike = ...,
        out: None = ...,
    ) -> Any: ...
    @overload
    def trace(
        self,
        offset: SupportsIndex = ...,
        axis1: SupportsIndex = ...,
        axis2: SupportsIndex = ...,
        dtype: DTypeLike = ...,
        out: _NdArraySubClass = ...,
    ) -> _NdArraySubClass: ...
    @overload
    def take(
        self: ndarray[Any, _dtype[_ScalarType]],
        indices: _IntLike_co,
        axis: Optional[SupportsIndex] = ...,
        out: None = ...,
        mode: _ModeKind = ...,
    ) -> _ScalarType: ...
    @overload
    def take(
        self,
        indices: _ArrayLikeInt_co,
        axis: Optional[SupportsIndex] = ...,
        out: None = ...,
        mode: _ModeKind = ...,
    ) -> ndarray[Any, _DType_co]: ...
    @overload
    def take(
        self,
        indices: _ArrayLikeInt_co,
        axis: Optional[SupportsIndex] = ...,
        out: _NdArraySubClass = ...,
        mode: _ModeKind = ...,
    ) -> _NdArraySubClass: ...
    def repeat(
        self, repeats: _ArrayLikeInt_co, axis: Optional[SupportsIndex] = ...
    ) -> ndarray[Any, _DType_co]: ...
    def flatten(self, order: _OrderKACF = ...) -> ndarray[Any, _DType_co]: ...
    def ravel(self, order: _OrderKACF = ...) -> ndarray[Any, _DType_co]: ...
    @overload
    def reshape(
        self, __shape: _ShapeLike, *, order: _OrderACF = ...
    ) -> ndarray[Any, _DType_co]: ...
    @overload
    def reshape(
        self, *shape: SupportsIndex, order: _OrderACF = ...
    ) -> ndarray[Any, _DType_co]: ...
    @overload
    def astype(
        self,
        dtype: _DTypeLike[_ScalarType],
        order: _OrderKACF = ...,
        casting: _Casting = ...,
        subok: bool = ...,
        copy: bool = ...,
    ) -> NDArray[_ScalarType]: ...
    @overload
    def astype(
        self,
        dtype: DTypeLike,
        order: _OrderKACF = ...,
        casting: _Casting = ...,
        subok: bool = ...,
        copy: bool = ...,
    ) -> NDArray[Any]: ...
    @overload
    def view(self: _ArraySelf) -> _ArraySelf: ...
    @overload
    def view(self, type: Type[_NdArraySubClass]) -> _NdArraySubClass: ...
    @overload
    def view(self, dtype: _DTypeLike[_ScalarType]) -> NDArray[_ScalarType]: ...
    @overload
    def view(self, dtype: DTypeLike) -> NDArray[Any]: ...
    @overload
    def view(
        self, dtype: DTypeLike, type: Type[_NdArraySubClass]
    ) -> _NdArraySubClass: ...
    @overload
    def getfield(
        self, dtype: _DTypeLike[_ScalarType], offset: SupportsIndex = ...
    ) -> NDArray[_ScalarType]: ...
    @overload
    def getfield(self, dtype: DTypeLike, offset: SupportsIndex = ...) -> NDArray[Any]: ...
    def __int__(self: ndarray[Any, _dtype[SupportsInt]]) -> int: ...
    def __float__(self: ndarray[Any, _dtype[SupportsFloat]]) -> float: ...
    def __complex__(self: ndarray[Any, _dtype[SupportsComplex]]) -> complex: ...
    def __index__(self: ndarray[Any, _dtype[SupportsIndex]]) -> int: ...
    def __len__(self) -> int: ...
    def __setitem__(self, key, value): ...
    def __iter__(self) -> Any: ...
    def __contains__(self, key) -> bool: ...
    @overload
    def __lt__(
        self: NDArray[Any], other: _NestedSequence[Union[str, bytes]]
    ) -> NoReturn: ...
    @overload
    def __lt__(self: _ArrayNumber_co, other: _ArrayLikeNumber_co) -> NDArray[bool_]: ...
    @overload
    def __lt__(self: _ArrayTD64_co, other: _ArrayLikeTD64_co) -> NDArray[bool_]: ...
    @overload
    def __lt__(self: NDArray[datetime64], other: _ArrayLikeDT64_co) -> NDArray[bool_]: ...
    @overload
    def __lt__(self: NDArray[object_], other: Any) -> NDArray[bool_]: ...
    @overload
    def __lt__(self: NDArray[Any], other: _ArrayLikeObject_co) -> NDArray[bool_]: ...
    @overload
    def __lt__(
        self: NDArray[Union[number[Any], datetime64, timedelta64, bool_]],
        other: _RecursiveSequence,
    ) -> NDArray[bool_]: ...
    @overload
    def __le__(
        self: NDArray[Any], other: _NestedSequence[Union[str, bytes]]
    ) -> NoReturn: ...
    @overload
    def __le__(self: _ArrayNumber_co, other: _ArrayLikeNumber_co) -> NDArray[bool_]: ...
    @overload
    def __le__(self: _ArrayTD64_co, other: _ArrayLikeTD64_co) -> NDArray[bool_]: ...
    @overload
    def __le__(self: NDArray[datetime64], other: _ArrayLikeDT64_co) -> NDArray[bool_]: ...
    @overload
    def __le__(self: NDArray[object_], other: Any) -> NDArray[bool_]: ...
    @overload
    def __le__(self: NDArray[Any], other: _ArrayLikeObject_co) -> NDArray[bool_]: ...
    @overload
    def __le__(
        self: NDArray[Union[number[Any], datetime64, timedelta64, bool_]],
        other: _RecursiveSequence,
    ) -> NDArray[bool_]: ...
    @overload
    def __gt__(
        self: NDArray[Any], other: _NestedSequence[Union[str, bytes]]
    ) -> NoReturn: ...
    @overload
    def __gt__(self: _ArrayNumber_co, other: _ArrayLikeNumber_co) -> NDArray[bool_]: ...
    @overload
    def __gt__(self: _ArrayTD64_co, other: _ArrayLikeTD64_co) -> NDArray[bool_]: ...
    @overload
    def __gt__(self: NDArray[datetime64], other: _ArrayLikeDT64_co) -> NDArray[bool_]: ...
    @overload
    def __gt__(self: NDArray[object_], other: Any) -> NDArray[bool_]: ...
    @overload
    def __gt__(self: NDArray[Any], other: _ArrayLikeObject_co) -> NDArray[bool_]: ...
    @overload
    def __gt__(
        self: NDArray[Union[number[Any], datetime64, timedelta64, bool_]],
        other: _RecursiveSequence,
    ) -> NDArray[bool_]: ...
    @overload
    def __ge__(
        self: NDArray[Any], other: _NestedSequence[Union[str, bytes]]
    ) -> NoReturn: ...
    @overload
    def __ge__(self: _ArrayNumber_co, other: _ArrayLikeNumber_co) -> NDArray[bool_]: ...
    @overload
    def __ge__(self: _ArrayTD64_co, other: _ArrayLikeTD64_co) -> NDArray[bool_]: ...
    @overload
    def __ge__(self: NDArray[datetime64], other: _ArrayLikeDT64_co) -> NDArray[bool_]: ...
    @overload
    def __ge__(self: NDArray[object_], other: Any) -> NDArray[bool_]: ...
    @overload
    def __ge__(self: NDArray[Any], other: _ArrayLikeObject_co) -> NDArray[bool_]: ...
    @overload
    def __ge__(
        self: NDArray[Union[number[Any], datetime64, timedelta64, bool_]],
        other: _RecursiveSequence,
    ) -> NDArray[bool_]: ...
    @overload
    def __abs__(self: NDArray[bool_]) -> NDArray[bool_]: ...
    @overload
    def __abs__(
        self: NDArray[complexfloating[_NBit1, _NBit1]]
    ) -> NDArray[floating[_NBit1]]: ...
    @overload
    def __abs__(self: NDArray[_NumberType]) -> NDArray[_NumberType]: ...
    @overload
    def __abs__(self: NDArray[timedelta64]) -> NDArray[timedelta64]: ...
    @overload
    def __abs__(self: NDArray[object_]) -> Any: ...
    @overload
    def __invert__(self: NDArray[bool_]) -> NDArray[bool_]: ...
    @overload
    def __invert__(self: NDArray[_IntType]) -> NDArray[_IntType]: ...
    @overload
    def __invert__(self: NDArray[object_]) -> Any: ...
    @overload
    def __pos__(self: NDArray[_NumberType]) -> NDArray[_NumberType]: ...
    @overload
    def __pos__(self: NDArray[timedelta64]) -> NDArray[timedelta64]: ...
    @overload
    def __pos__(self: NDArray[object_]) -> Any: ...
    @overload
    def __neg__(self: NDArray[_NumberType]) -> NDArray[_NumberType]: ...
    @overload
    def __neg__(self: NDArray[timedelta64]) -> NDArray[timedelta64]: ...
    @overload
    def __neg__(self: NDArray[object_]) -> Any: ...
    @overload
    def __matmul__(
        self: NDArray[Any], other: _NestedSequence[Union[str, bytes]]
    ) -> NoReturn: ...
    @overload
    def __matmul__(self: NDArray[bool_], other: _ArrayLikeBool_co) -> NDArray[bool_]: ...
    @overload
    def __matmul__(
        self: _ArrayUInt_co, other: _ArrayLikeUInt_co
    ) -> NDArray[unsignedinteger[Any]]: ...
    @overload
    def __matmul__(
        self: _ArrayInt_co, other: _ArrayLikeInt_co
    ) -> NDArray[signedinteger[Any]]: ...
    @overload
    def __matmul__(
        self: _ArrayFloat_co, other: _ArrayLikeFloat_co
    ) -> NDArray[floating[Any]]: ...
    @overload
    def __matmul__(
        self: _ArrayComplex_co, other: _ArrayLikeComplex_co
    ) -> NDArray[complexfloating[Any, Any]]: ...
    @overload
    def __matmul__(self: NDArray[object_], other: Any) -> Any: ...
    @overload
    def __matmul__(self: NDArray[Any], other: _ArrayLikeObject_co) -> Any: ...
    @overload
    def __matmul__(self: _ArrayNumber_co, other: _RecursiveSequence) -> Any: ...
    @overload
    def __rmatmul__(
        self: NDArray[Any], other: _NestedSequence[Union[str, bytes]]
    ) -> NoReturn: ...
    @overload
    def __rmatmul__(self: NDArray[bool_], other: _ArrayLikeBool_co) -> NDArray[bool_]: ...
    @overload
    def __rmatmul__(
        self: _ArrayUInt_co, other: _ArrayLikeUInt_co
    ) -> NDArray[unsignedinteger[Any]]: ...
    @overload
    def __rmatmul__(
        self: _ArrayInt_co, other: _ArrayLikeInt_co
    ) -> NDArray[signedinteger[Any]]: ...
    @overload
    def __rmatmul__(
        self: _ArrayFloat_co, other: _ArrayLikeFloat_co
    ) -> NDArray[floating[Any]]: ...
    @overload
    def __rmatmul__(
        self: _ArrayComplex_co, other: _ArrayLikeComplex_co
    ) -> NDArray[complexfloating[Any, Any]]: ...
    @overload
    def __rmatmul__(self: NDArray[object_], other: Any) -> Any: ...
    @overload
    def __rmatmul__(self: NDArray[Any], other: _ArrayLikeObject_co) -> Any: ...
    @overload
    def __rmatmul__(self: _ArrayNumber_co, other: _RecursiveSequence) -> Any: ...
    @overload
    def __mod__(
        self: NDArray[Any], other: _NestedSequence[Union[str, bytes]]
    ) -> NoReturn: ...
    @overload
    def __mod__(self: NDArray[bool_], other: _ArrayLikeBool_co) -> NDArray[int8]: ...
    @overload
    def __mod__(
        self: _ArrayUInt_co, other: _ArrayLikeUInt_co
    ) -> NDArray[unsignedinteger[Any]]: ...
    @overload
    def __mod__(
        self: _ArrayInt_co, other: _ArrayLikeInt_co
    ) -> NDArray[signedinteger[Any]]: ...
    @overload
    def __mod__(
        self: _ArrayFloat_co, other: _ArrayLikeFloat_co
    ) -> NDArray[floating[Any]]: ...
    @overload
    def __mod__(
        self: _ArrayTD64_co, other: _NestedSequence[_SupportsArray[_dtype[timedelta64]]]
    ) -> NDArray[timedelta64]: ...
    @overload
    def __mod__(self: NDArray[object_], other: Any) -> Any: ...
    @overload
    def __mod__(self: NDArray[Any], other: _ArrayLikeObject_co) -> Any: ...
    @overload
    def __mod__(
        self: NDArray[Union[bool_, integer[Any], floating[Any], timedelta64]],
        other: _RecursiveSequence,
    ) -> Any: ...
    @overload
    def __rmod__(
        self: NDArray[Any], other: _NestedSequence[Union[str, bytes]]
    ) -> NoReturn: ...
    @overload
    def __rmod__(self: NDArray[bool_], other: _ArrayLikeBool_co) -> NDArray[int8]: ...
    @overload
    def __rmod__(
        self: _ArrayUInt_co, other: _ArrayLikeUInt_co
    ) -> NDArray[unsignedinteger[Any]]: ...
    @overload
    def __rmod__(
        self: _ArrayInt_co, other: _ArrayLikeInt_co
    ) -> NDArray[signedinteger[Any]]: ...
    @overload
    def __rmod__(
        self: _ArrayFloat_co, other: _ArrayLikeFloat_co
    ) -> NDArray[floating[Any]]: ...
    @overload
    def __rmod__(
        self: _ArrayTD64_co, other: _NestedSequence[_SupportsArray[_dtype[timedelta64]]]
    ) -> NDArray[timedelta64]: ...
    @overload
    def __rmod__(self: NDArray[object_], other: Any) -> Any: ...
    @overload
    def __rmod__(self: NDArray[Any], other: _ArrayLikeObject_co) -> Any: ...
    @overload
    def __rmod__(
        self: NDArray[Union[bool_, integer[Any], floating[Any], timedelta64]],
        other: _RecursiveSequence,
    ) -> Any: ...
    @overload
    def __divmod__(
        self: NDArray[Any], other: _NestedSequence[Union[str, bytes]]
    ) -> NoReturn: ...
    @overload
    def __divmod__(
        self: NDArray[bool_], other: _ArrayLikeBool_co
    ) -> _2Tuple[NDArray[int8]]: ...
    @overload
    def __divmod__(
        self: _ArrayUInt_co, other: _ArrayLikeUInt_co
    ) -> _2Tuple[NDArray[unsignedinteger[Any]]]: ...
    @overload
    def __divmod__(
        self: _ArrayInt_co, other: _ArrayLikeInt_co
    ) -> _2Tuple[NDArray[signedinteger[Any]]]: ...
    @overload
    def __divmod__(
        self: _ArrayFloat_co, other: _ArrayLikeFloat_co
    ) -> _2Tuple[NDArray[floating[Any]]]: ...
    @overload
    def __divmod__(
        self: _ArrayTD64_co, other: _NestedSequence[_SupportsArray[_dtype[timedelta64]]]
    ) -> Tuple[NDArray[int64], NDArray[timedelta64]]: ...
    @overload
    def __divmod__(
        self: NDArray[Union[bool_, integer[Any], floating[Any], timedelta64]],
        other: _RecursiveSequence,
    ) -> _2Tuple[Any]: ...
    @overload
    def __rdivmod__(
        self: NDArray[Any], other: _NestedSequence[Union[str, bytes]]
    ) -> NoReturn: ...
    @overload
    def __rdivmod__(
        self: NDArray[bool_], other: _ArrayLikeBool_co
    ) -> _2Tuple[NDArray[int8]]: ...
    @overload
    def __rdivmod__(
        self: _ArrayUInt_co, other: _ArrayLikeUInt_co
    ) -> _2Tuple[NDArray[unsignedinteger[Any]]]: ...
    @overload
    def __rdivmod__(
        self: _ArrayInt_co, other: _ArrayLikeInt_co
    ) -> _2Tuple[NDArray[signedinteger[Any]]]: ...
    @overload
    def __rdivmod__(
        self: _ArrayFloat_co, other: _ArrayLikeFloat_co
    ) -> _2Tuple[NDArray[floating[Any]]]: ...
    @overload
    def __rdivmod__(
        self: _ArrayTD64_co, other: _NestedSequence[_SupportsArray[_dtype[timedelta64]]]
    ) -> Tuple[NDArray[int64], NDArray[timedelta64]]: ...
    @overload
    def __rdivmod__(
        self: NDArray[Union[bool_, integer[Any], floating[Any], timedelta64]],
        other: _RecursiveSequence,
    ) -> _2Tuple[Any]: ...
    @overload
    def __add__(
        self: NDArray[Any], other: _NestedSequence[Union[str, bytes]]
    ) -> NoReturn: ...
    @overload
    def __add__(self: NDArray[bool_], other: _ArrayLikeBool_co) -> NDArray[bool_]: ...
    @overload
    def __add__(
        self: _ArrayUInt_co, other: _ArrayLikeUInt_co
    ) -> NDArray[unsignedinteger[Any]]: ...
    @overload
    def __add__(
        self: _ArrayInt_co, other: _ArrayLikeInt_co
    ) -> NDArray[signedinteger[Any]]: ...
    @overload
    def __add__(
        self: _ArrayFloat_co, other: _ArrayLikeFloat_co
    ) -> NDArray[floating[Any]]: ...
    @overload
    def __add__(
        self: _ArrayComplex_co, other: _ArrayLikeComplex_co
    ) -> NDArray[complexfloating[Any, Any]]: ...
    @overload
    def __add__(
        self: _ArrayTD64_co, other: _ArrayLikeTD64_co
    ) -> NDArray[timedelta64]: ...
    @overload
    def __add__(self: _ArrayTD64_co, other: _ArrayLikeDT64_co) -> NDArray[datetime64]: ...
    @overload
    def __add__(
        self: NDArray[datetime64], other: _ArrayLikeTD64_co
    ) -> NDArray[datetime64]: ...
    @overload
    def __add__(self: NDArray[object_], other: Any) -> Any: ...
    @overload
    def __add__(self: NDArray[Any], other: _ArrayLikeObject_co) -> Any: ...
    @overload
    def __add__(
        self: NDArray[Union[bool_, number[Any], timedelta64, datetime64]],
        other: _RecursiveSequence,
    ) -> Any: ...
    @overload
    def __radd__(
        self: NDArray[Any], other: _NestedSequence[Union[str, bytes]]
    ) -> NoReturn: ...
    @overload
    def __radd__(self: NDArray[bool_], other: _ArrayLikeBool_co) -> NDArray[bool_]: ...
    @overload
    def __radd__(
        self: _ArrayUInt_co, other: _ArrayLikeUInt_co
    ) -> NDArray[unsignedinteger[Any]]: ...
    @overload
    def __radd__(
        self: _ArrayInt_co, other: _ArrayLikeInt_co
    ) -> NDArray[signedinteger[Any]]: ...
    @overload
    def __radd__(
        self: _ArrayFloat_co, other: _ArrayLikeFloat_co
    ) -> NDArray[floating[Any]]: ...
    @overload
    def __radd__(
        self: _ArrayComplex_co, other: _ArrayLikeComplex_co
    ) -> NDArray[complexfloating[Any, Any]]: ...
    @overload
    def __radd__(
        self: _ArrayTD64_co, other: _ArrayLikeTD64_co
    ) -> NDArray[timedelta64]: ...
    @overload
    def __radd__(
        self: _ArrayTD64_co, other: _ArrayLikeDT64_co
    ) -> NDArray[datetime64]: ...
    @overload
    def __radd__(
        self: NDArray[datetime64], other: _ArrayLikeTD64_co
    ) -> NDArray[datetime64]: ...
    @overload
    def __radd__(self: NDArray[object_], other: Any) -> Any: ...
    @overload
    def __radd__(self: NDArray[Any], other: _ArrayLikeObject_co) -> Any: ...
    @overload
    def __radd__(
        self: NDArray[Union[bool_, number[Any], timedelta64, datetime64]],
        other: _RecursiveSequence,
    ) -> Any: ...
    @overload
    def __sub__(
        self: NDArray[Any], other: _NestedSequence[Union[str, bytes]]
    ) -> NoReturn: ...
    @overload
    def __sub__(self: NDArray[bool_], other: _ArrayLikeBool_co) -> NoReturn: ...
    @overload
    def __sub__(
        self: _ArrayUInt_co, other: _ArrayLikeUInt_co
    ) -> NDArray[unsignedinteger[Any]]: ...
    @overload
    def __sub__(
        self: _ArrayInt_co, other: _ArrayLikeInt_co
    ) -> NDArray[signedinteger[Any]]: ...
    @overload
    def __sub__(
        self: _ArrayFloat_co, other: _ArrayLikeFloat_co
    ) -> NDArray[floating[Any]]: ...
    @overload
    def __sub__(
        self: _ArrayComplex_co, other: _ArrayLikeComplex_co
    ) -> NDArray[complexfloating[Any, Any]]: ...
    @overload
    def __sub__(
        self: _ArrayTD64_co, other: _ArrayLikeTD64_co
    ) -> NDArray[timedelta64]: ...
    @overload
    def __sub__(
        self: NDArray[datetime64], other: _ArrayLikeTD64_co
    ) -> NDArray[datetime64]: ...
    @overload
    def __sub__(
        self: NDArray[datetime64], other: _ArrayLikeDT64_co
    ) -> NDArray[timedelta64]: ...
    @overload
    def __sub__(self: NDArray[object_], other: Any) -> Any: ...
    @overload
    def __sub__(self: NDArray[Any], other: _ArrayLikeObject_co) -> Any: ...
    @overload
    def __sub__(
        self: NDArray[Union[bool_, number[Any], timedelta64, datetime64]],
        other: _RecursiveSequence,
    ) -> Any: ...
    @overload
    def __rsub__(
        self: NDArray[Any], other: _NestedSequence[Union[str, bytes]]
    ) -> NoReturn: ...
    @overload
    def __rsub__(self: NDArray[bool_], other: _ArrayLikeBool_co) -> NoReturn: ...
    @overload
    def __rsub__(
        self: _ArrayUInt_co, other: _ArrayLikeUInt_co
    ) -> NDArray[unsignedinteger[Any]]: ...
    @overload
    def __rsub__(
        self: _ArrayInt_co, other: _ArrayLikeInt_co
    ) -> NDArray[signedinteger[Any]]: ...
    @overload
    def __rsub__(
        self: _ArrayFloat_co, other: _ArrayLikeFloat_co
    ) -> NDArray[floating[Any]]: ...
    @overload
    def __rsub__(
        self: _ArrayComplex_co, other: _ArrayLikeComplex_co
    ) -> NDArray[complexfloating[Any, Any]]: ...
    @overload
    def __rsub__(
        self: _ArrayTD64_co, other: _ArrayLikeTD64_co
    ) -> NDArray[timedelta64]: ...
    @overload
    def __rsub__(
        self: _ArrayTD64_co, other: _ArrayLikeDT64_co
    ) -> NDArray[datetime64]: ...
    @overload
    def __rsub__(
        self: NDArray[datetime64], other: _ArrayLikeDT64_co
    ) -> NDArray[timedelta64]: ...
    @overload
    def __rsub__(self: NDArray[object_], other: Any) -> Any: ...
    @overload
    def __rsub__(self: NDArray[Any], other: _ArrayLikeObject_co) -> Any: ...
    @overload
    def __rsub__(
        self: NDArray[Union[bool_, number[Any], timedelta64, datetime64]],
        other: _RecursiveSequence,
    ) -> Any: ...
    @overload
    def __mul__(
        self: NDArray[Any], other: _NestedSequence[Union[str, bytes]]
    ) -> NoReturn: ...
    @overload
    def __mul__(self: NDArray[bool_], other: _ArrayLikeBool_co) -> NDArray[bool_]: ...
    @overload
    def __mul__(
        self: _ArrayUInt_co, other: _ArrayLikeUInt_co
    ) -> NDArray[unsignedinteger[Any]]: ...
    @overload
    def __mul__(
        self: _ArrayInt_co, other: _ArrayLikeInt_co
    ) -> NDArray[signedinteger[Any]]: ...
    @overload
    def __mul__(
        self: _ArrayFloat_co, other: _ArrayLikeFloat_co
    ) -> NDArray[floating[Any]]: ...
    @overload
    def __mul__(
        self: _ArrayComplex_co, other: _ArrayLikeComplex_co
    ) -> NDArray[complexfloating[Any, Any]]: ...
    @overload
    def __mul__(
        self: _ArrayTD64_co, other: _ArrayLikeFloat_co
    ) -> NDArray[timedelta64]: ...
    @overload
    def __mul__(
        self: _ArrayFloat_co, other: _ArrayLikeTD64_co
    ) -> NDArray[timedelta64]: ...
    @overload
    def __mul__(self: NDArray[object_], other: Any) -> Any: ...
    @overload
    def __mul__(self: NDArray[Any], other: _ArrayLikeObject_co) -> Any: ...
    @overload
    def __mul__(
        self: NDArray[Union[bool_, number[Any], timedelta64]], other: _RecursiveSequence
    ) -> Any: ...
    @overload
    def __rmul__(
        self: NDArray[Any], other: _NestedSequence[Union[str, bytes]]
    ) -> NoReturn: ...
    @overload
    def __rmul__(self: NDArray[bool_], other: _ArrayLikeBool_co) -> NDArray[bool_]: ...
    @overload
    def __rmul__(
        self: _ArrayUInt_co, other: _ArrayLikeUInt_co
    ) -> NDArray[unsignedinteger[Any]]: ...
    @overload
    def __rmul__(
        self: _ArrayInt_co, other: _ArrayLikeInt_co
    ) -> NDArray[signedinteger[Any]]: ...
    @overload
    def __rmul__(
        self: _ArrayFloat_co, other: _ArrayLikeFloat_co
    ) -> NDArray[floating[Any]]: ...
    @overload
    def __rmul__(
        self: _ArrayComplex_co, other: _ArrayLikeComplex_co
    ) -> NDArray[complexfloating[Any, Any]]: ...
    @overload
    def __rmul__(
        self: _ArrayTD64_co, other: _ArrayLikeFloat_co
    ) -> NDArray[timedelta64]: ...
    @overload
    def __rmul__(
        self: _ArrayFloat_co, other: _ArrayLikeTD64_co
    ) -> NDArray[timedelta64]: ...
    @overload
    def __rmul__(self: NDArray[object_], other: Any) -> Any: ...
    @overload
    def __rmul__(self: NDArray[Any], other: _ArrayLikeObject_co) -> Any: ...
    @overload
    def __rmul__(
        self: NDArray[Union[bool_, number[Any], timedelta64]], other: _RecursiveSequence
    ) -> Any: ...
    @overload
    def __floordiv__(
        self: NDArray[Any], other: _NestedSequence[Union[str, bytes]]
    ) -> NoReturn: ...
    @overload
    def __floordiv__(self: NDArray[bool_], other: _ArrayLikeBool_co) -> NDArray[int8]: ...
    @overload
    def __floordiv__(
        self: _ArrayUInt_co, other: _ArrayLikeUInt_co
    ) -> NDArray[unsignedinteger[Any]]: ...
    @overload
    def __floordiv__(
        self: _ArrayInt_co, other: _ArrayLikeInt_co
    ) -> NDArray[signedinteger[Any]]: ...
    @overload
    def __floordiv__(
        self: _ArrayFloat_co, other: _ArrayLikeFloat_co
    ) -> NDArray[floating[Any]]: ...
    @overload
    def __floordiv__(
        self: _ArrayComplex_co, other: _ArrayLikeComplex_co
    ) -> NDArray[complexfloating[Any, Any]]: ...
    @overload
    def __floordiv__(
        self: NDArray[timedelta64],
        other: _NestedSequence[_SupportsArray[_dtype[timedelta64]]],
    ) -> NDArray[int64]: ...
    @overload
    def __floordiv__(
        self: NDArray[timedelta64], other: _ArrayLikeBool_co
    ) -> NoReturn: ...
    @overload
    def __floordiv__(
        self: NDArray[timedelta64], other: _ArrayLikeFloat_co
    ) -> NDArray[timedelta64]: ...
    @overload
    def __floordiv__(self: NDArray[object_], other: Any) -> Any: ...
    @overload
    def __floordiv__(self: NDArray[Any], other: _ArrayLikeObject_co) -> Any: ...
    @overload
    def __floordiv__(
        self: NDArray[Union[bool_, number[Any], timedelta64]], other: _RecursiveSequence
    ) -> Any: ...
    @overload
    def __rfloordiv__(
        self: NDArray[Any], other: _NestedSequence[Union[str, bytes]]
    ) -> NoReturn: ...
    @overload
    def __rfloordiv__(
        self: NDArray[bool_], other: _ArrayLikeBool_co
    ) -> NDArray[int8]: ...
    @overload
    def __rfloordiv__(
        self: _ArrayUInt_co, other: _ArrayLikeUInt_co
    ) -> NDArray[unsignedinteger[Any]]: ...
    @overload
    def __rfloordiv__(
        self: _ArrayInt_co, other: _ArrayLikeInt_co
    ) -> NDArray[signedinteger[Any]]: ...
    @overload
    def __rfloordiv__(
        self: _ArrayFloat_co, other: _ArrayLikeFloat_co
    ) -> NDArray[floating[Any]]: ...
    @overload
    def __rfloordiv__(
        self: _ArrayComplex_co, other: _ArrayLikeComplex_co
    ) -> NDArray[complexfloating[Any, Any]]: ...
    @overload
    def __rfloordiv__(
        self: NDArray[timedelta64],
        other: _NestedSequence[_SupportsArray[_dtype[timedelta64]]],
    ) -> NDArray[int64]: ...
    @overload
    def __rfloordiv__(self: NDArray[bool_], other: _ArrayLikeTD64_co) -> NoReturn: ...
    @overload
    def __rfloordiv__(
        self: _ArrayFloat_co, other: _ArrayLikeTD64_co
    ) -> NDArray[timedelta64]: ...
    @overload
    def __rfloordiv__(self: NDArray[object_], other: Any) -> Any: ...
    @overload
    def __rfloordiv__(self: NDArray[Any], other: _ArrayLikeObject_co) -> Any: ...
    @overload
    def __rfloordiv__(
        self: NDArray[Union[bool_, number[Any], timedelta64]], other: _RecursiveSequence
    ) -> Any: ...
    @overload
    def __pow__(
        self: NDArray[Any], other: _NestedSequence[Union[str, bytes]]
    ) -> NoReturn: ...
    @overload
    def __pow__(self: NDArray[bool_], other: _ArrayLikeBool_co) -> NDArray[int8]: ...
    @overload
    def __pow__(
        self: _ArrayUInt_co, other: _ArrayLikeUInt_co
    ) -> NDArray[unsignedinteger[Any]]: ...
    @overload
    def __pow__(
        self: _ArrayInt_co, other: _ArrayLikeInt_co
    ) -> NDArray[signedinteger[Any]]: ...
    @overload
    def __pow__(
        self: _ArrayFloat_co, other: _ArrayLikeFloat_co
    ) -> NDArray[floating[Any]]: ...
    @overload
    def __pow__(
        self: _ArrayComplex_co, other: _ArrayLikeComplex_co
    ) -> NDArray[complexfloating[Any, Any]]: ...
    @overload
    def __pow__(self: NDArray[object_], other: Any) -> Any: ...
    @overload
    def __pow__(self: NDArray[Any], other: _ArrayLikeObject_co) -> Any: ...
    @overload
    def __pow__(
        self: NDArray[Union[bool_, number[Any]]], other: _RecursiveSequence
    ) -> Any: ...
    @overload
    def __rpow__(
        self: NDArray[Any], other: _NestedSequence[Union[str, bytes]]
    ) -> NoReturn: ...
    @overload
    def __rpow__(self: NDArray[bool_], other: _ArrayLikeBool_co) -> NDArray[int8]: ...
    @overload
    def __rpow__(
        self: _ArrayUInt_co, other: _ArrayLikeUInt_co
    ) -> NDArray[unsignedinteger[Any]]: ...
    @overload
    def __rpow__(
        self: _ArrayInt_co, other: _ArrayLikeInt_co
    ) -> NDArray[signedinteger[Any]]: ...
    @overload
    def __rpow__(
        self: _ArrayFloat_co, other: _ArrayLikeFloat_co
    ) -> NDArray[floating[Any]]: ...
    @overload
    def __rpow__(
        self: _ArrayComplex_co, other: _ArrayLikeComplex_co
    ) -> NDArray[complexfloating[Any, Any]]: ...
    @overload
    def __rpow__(self: NDArray[object_], other: Any) -> Any: ...
    @overload
    def __rpow__(self: NDArray[Any], other: _ArrayLikeObject_co) -> Any: ...
    @overload
    def __rpow__(
        self: NDArray[Union[bool_, number[Any]]], other: _RecursiveSequence
    ) -> Any: ...
    @overload
    def __truediv__(
        self: NDArray[Any], other: _NestedSequence[Union[str, bytes]]
    ) -> NoReturn: ...
    @overload
    def __truediv__(self: _ArrayInt_co, other: _ArrayInt_co) -> NDArray[float64]: ...
    @overload
    def __truediv__(
        self: _ArrayFloat_co, other: _ArrayLikeFloat_co
    ) -> NDArray[floating[Any]]: ...
    @overload
    def __truediv__(
        self: _ArrayComplex_co, other: _ArrayLikeComplex_co
    ) -> NDArray[complexfloating[Any, Any]]: ...
    @overload
    def __truediv__(
        self: NDArray[timedelta64],
        other: _NestedSequence[_SupportsArray[_dtype[timedelta64]]],
    ) -> NDArray[float64]: ...
    @overload
    def __truediv__(self: NDArray[timedelta64], other: _ArrayLikeBool_co) -> NoReturn: ...
    @overload
    def __truediv__(
        self: NDArray[timedelta64], other: _ArrayLikeFloat_co
    ) -> NDArray[timedelta64]: ...
    @overload
    def __truediv__(self: NDArray[object_], other: Any) -> Any: ...
    @overload
    def __truediv__(self: NDArray[Any], other: _ArrayLikeObject_co) -> Any: ...
    @overload
    def __truediv__(
        self: NDArray[Union[bool_, number[Any], timedelta64]], other: _RecursiveSequence
    ) -> Any: ...
    @overload
    def __rtruediv__(
        self: NDArray[Any], other: _NestedSequence[Union[str, bytes]]
    ) -> NoReturn: ...
    @overload
    def __rtruediv__(self: _ArrayInt_co, other: _ArrayInt_co) -> NDArray[float64]: ...
    @overload
    def __rtruediv__(
        self: _ArrayFloat_co, other: _ArrayLikeFloat_co
    ) -> NDArray[floating[Any]]: ...
    @overload
    def __rtruediv__(
        self: _ArrayComplex_co, other: _ArrayLikeComplex_co
    ) -> NDArray[complexfloating[Any, Any]]: ...
    @overload
    def __rtruediv__(
        self: NDArray[timedelta64],
        other: _NestedSequence[_SupportsArray[_dtype[timedelta64]]],
    ) -> NDArray[float64]: ...
    @overload
    def __rtruediv__(self: NDArray[bool_], other: _ArrayLikeTD64_co) -> NoReturn: ...
    @overload
    def __rtruediv__(
        self: _ArrayFloat_co, other: _ArrayLikeTD64_co
    ) -> NDArray[timedelta64]: ...
    @overload
    def __rtruediv__(self: NDArray[object_], other: Any) -> Any: ...
    @overload
    def __rtruediv__(self: NDArray[Any], other: _ArrayLikeObject_co) -> Any: ...
    @overload
    def __rtruediv__(
        self: NDArray[Union[bool_, number[Any], timedelta64]], other: _RecursiveSequence
    ) -> Any: ...
    @overload
    def __lshift__(
        self: NDArray[Any], other: _NestedSequence[Union[str, bytes]]
    ) -> NoReturn: ...
    @overload
    def __lshift__(self: NDArray[bool_], other: _ArrayLikeBool_co) -> NDArray[int8]: ...
    @overload
    def __lshift__(
        self: _ArrayUInt_co, other: _ArrayLikeUInt_co
    ) -> NDArray[unsignedinteger[Any]]: ...
    @overload
    def __lshift__(
        self: _ArrayInt_co, other: _ArrayLikeInt_co
    ) -> NDArray[signedinteger[Any]]: ...
    @overload
    def __lshift__(self: NDArray[object_], other: Any) -> Any: ...
    @overload
    def __lshift__(self: NDArray[Any], other: _ArrayLikeObject_co) -> Any: ...
    @overload
    def __lshift__(
        self: NDArray[Union[bool_, integer[Any]]], other: _RecursiveSequence
    ) -> Any: ...
    @overload
    def __rlshift__(
        self: NDArray[Any], other: _NestedSequence[Union[str, bytes]]
    ) -> NoReturn: ...
    @overload
    def __rlshift__(self: NDArray[bool_], other: _ArrayLikeBool_co) -> NDArray[int8]: ...
    @overload
    def __rlshift__(
        self: _ArrayUInt_co, other: _ArrayLikeUInt_co
    ) -> NDArray[unsignedinteger[Any]]: ...
    @overload
    def __rlshift__(
        self: _ArrayInt_co, other: _ArrayLikeInt_co
    ) -> NDArray[signedinteger[Any]]: ...
    @overload
    def __rlshift__(self: NDArray[object_], other: Any) -> Any: ...
    @overload
    def __rlshift__(self: NDArray[Any], other: _ArrayLikeObject_co) -> Any: ...
    @overload
    def __rlshift__(
        self: NDArray[Union[bool_, integer[Any]]], other: _RecursiveSequence
    ) -> Any: ...
    @overload
    def __rshift__(
        self: NDArray[Any], other: _NestedSequence[Union[str, bytes]]
    ) -> NoReturn: ...
    @overload
    def __rshift__(self: NDArray[bool_], other: _ArrayLikeBool_co) -> NDArray[int8]: ...
    @overload
    def __rshift__(
        self: _ArrayUInt_co, other: _ArrayLikeUInt_co
    ) -> NDArray[unsignedinteger[Any]]: ...
    @overload
    def __rshift__(
        self: _ArrayInt_co, other: _ArrayLikeInt_co
    ) -> NDArray[signedinteger[Any]]: ...
    @overload
    def __rshift__(self: NDArray[object_], other: Any) -> Any: ...
    @overload
    def __rshift__(self: NDArray[Any], other: _ArrayLikeObject_co) -> Any: ...
    @overload
    def __rshift__(
        self: NDArray[Union[bool_, integer[Any]]], other: _RecursiveSequence
    ) -> Any: ...
    @overload
    def __rrshift__(
        self: NDArray[Any], other: _NestedSequence[Union[str, bytes]]
    ) -> NoReturn: ...
    @overload
    def __rrshift__(self: NDArray[bool_], other: _ArrayLikeBool_co) -> NDArray[int8]: ...
    @overload
    def __rrshift__(
        self: _ArrayUInt_co, other: _ArrayLikeUInt_co
    ) -> NDArray[unsignedinteger[Any]]: ...
    @overload
    def __rrshift__(
        self: _ArrayInt_co, other: _ArrayLikeInt_co
    ) -> NDArray[signedinteger[Any]]: ...
    @overload
    def __rrshift__(self: NDArray[object_], other: Any) -> Any: ...
    @overload
    def __rrshift__(self: NDArray[Any], other: _ArrayLikeObject_co) -> Any: ...
    @overload
    def __rrshift__(
        self: NDArray[Union[bool_, integer[Any]]], other: _RecursiveSequence
    ) -> Any: ...
    @overload
    def __and__(
        self: NDArray[Any], other: _NestedSequence[Union[str, bytes]]
    ) -> NoReturn: ...
    @overload
    def __and__(self: NDArray[bool_], other: _ArrayLikeBool_co) -> NDArray[bool_]: ...
    @overload
    def __and__(
        self: _ArrayUInt_co, other: _ArrayLikeUInt_co
    ) -> NDArray[unsignedinteger[Any]]: ...
    @overload
    def __and__(
        self: _ArrayInt_co, other: _ArrayLikeInt_co
    ) -> NDArray[signedinteger[Any]]: ...
    @overload
    def __and__(self: NDArray[object_], other: Any) -> Any: ...
    @overload
    def __and__(self: NDArray[Any], other: _ArrayLikeObject_co) -> Any: ...
    @overload
    def __and__(
        self: NDArray[Union[bool_, integer[Any]]], other: _RecursiveSequence
    ) -> Any: ...
    @overload
    def __rand__(
        self: NDArray[Any], other: _NestedSequence[Union[str, bytes]]
    ) -> NoReturn: ...
    @overload
    def __rand__(self: NDArray[bool_], other: _ArrayLikeBool_co) -> NDArray[bool_]: ...
    @overload
    def __rand__(
        self: _ArrayUInt_co, other: _ArrayLikeUInt_co
    ) -> NDArray[unsignedinteger[Any]]: ...
    @overload
    def __rand__(
        self: _ArrayInt_co, other: _ArrayLikeInt_co
    ) -> NDArray[signedinteger[Any]]: ...
    @overload
    def __rand__(self: NDArray[object_], other: Any) -> Any: ...
    @overload
    def __rand__(self: NDArray[Any], other: _ArrayLikeObject_co) -> Any: ...
    @overload
    def __rand__(
        self: NDArray[Union[bool_, integer[Any]]], other: _RecursiveSequence
    ) -> Any: ...
    @overload
    def __xor__(
        self: NDArray[Any], other: _NestedSequence[Union[str, bytes]]
    ) -> NoReturn: ...
    @overload
    def __xor__(self: NDArray[bool_], other: _ArrayLikeBool_co) -> NDArray[bool_]: ...
    @overload
    def __xor__(
        self: _ArrayUInt_co, other: _ArrayLikeUInt_co
    ) -> NDArray[unsignedinteger[Any]]: ...
    @overload
    def __xor__(
        self: _ArrayInt_co, other: _ArrayLikeInt_co
    ) -> NDArray[signedinteger[Any]]: ...
    @overload
    def __xor__(self: NDArray[object_], other: Any) -> Any: ...
    @overload
    def __xor__(self: NDArray[Any], other: _ArrayLikeObject_co) -> Any: ...
    @overload
    def __xor__(
        self: NDArray[Union[bool_, integer[Any]]], other: _RecursiveSequence
    ) -> Any: ...
    @overload
    def __rxor__(
        self: NDArray[Any], other: _NestedSequence[Union[str, bytes]]
    ) -> NoReturn: ...
    @overload
    def __rxor__(self: NDArray[bool_], other: _ArrayLikeBool_co) -> NDArray[bool_]: ...
    @overload
    def __rxor__(
        self: _ArrayUInt_co, other: _ArrayLikeUInt_co
    ) -> NDArray[unsignedinteger[Any]]: ...
    @overload
    def __rxor__(
        self: _ArrayInt_co, other: _ArrayLikeInt_co
    ) -> NDArray[signedinteger[Any]]: ...
    @overload
    def __rxor__(self: NDArray[object_], other: Any) -> Any: ...
    @overload
    def __rxor__(self: NDArray[Any], other: _ArrayLikeObject_co) -> Any: ...
    @overload
    def __rxor__(
        self: NDArray[Union[bool_, integer[Any]]], other: _RecursiveSequence
    ) -> Any: ...
    @overload
    def __or__(
        self: NDArray[Any], other: _NestedSequence[Union[str, bytes]]
    ) -> NoReturn: ...
    @overload
    def __or__(self: NDArray[bool_], other: _ArrayLikeBool_co) -> NDArray[bool_]: ...
    @overload
    def __or__(
        self: _ArrayUInt_co, other: _ArrayLikeUInt_co
    ) -> NDArray[unsignedinteger[Any]]: ...
    @overload
    def __or__(
        self: _ArrayInt_co, other: _ArrayLikeInt_co
    ) -> NDArray[signedinteger[Any]]: ...
    @overload
    def __or__(self: NDArray[object_], other: Any) -> Any: ...
    @overload
    def __or__(self: NDArray[Any], other: _ArrayLikeObject_co) -> Any: ...
    @overload
    def __or__(
        self: NDArray[Union[bool_, integer[Any]]], other: _RecursiveSequence
    ) -> Any: ...
    @overload
    def __ror__(
        self: NDArray[Any], other: _NestedSequence[Union[str, bytes]]
    ) -> NoReturn: ...
    @overload
    def __ror__(self: NDArray[bool_], other: _ArrayLikeBool_co) -> NDArray[bool_]: ...
    @overload
    def __ror__(
        self: _ArrayUInt_co, other: _ArrayLikeUInt_co
    ) -> NDArray[unsignedinteger[Any]]: ...
    @overload
    def __ror__(
        self: _ArrayInt_co, other: _ArrayLikeInt_co
    ) -> NDArray[signedinteger[Any]]: ...
    @overload
    def __ror__(self: NDArray[object_], other: Any) -> Any: ...
    @overload
    def __ror__(self: NDArray[Any], other: _ArrayLikeObject_co) -> Any: ...
    @overload
    def __ror__(
        self: NDArray[Union[bool_, integer[Any]]], other: _RecursiveSequence
    ) -> Any: ...
    @overload
    def __iadd__(
        self: NDArray[Any], other: _NestedSequence[Union[str, bytes]]
    ) -> NoReturn: ...
    @overload
    def __iadd__(self: NDArray[bool_], other: _ArrayLikeBool_co) -> NDArray[bool_]: ...
    @overload
    def __iadd__(
        self: NDArray[unsignedinteger[_NBit1]], other: _ArrayLikeUInt_co
    ) -> NDArray[unsignedinteger[_NBit1]]: ...
    @overload
    def __iadd__(
        self: NDArray[signedinteger[_NBit1]], other: _ArrayLikeInt_co
    ) -> NDArray[signedinteger[_NBit1]]: ...
    @overload
    def __iadd__(
        self: NDArray[floating[_NBit1]], other: _ArrayLikeFloat_co
    ) -> NDArray[floating[_NBit1]]: ...
    @overload
    def __iadd__(
        self: NDArray[complexfloating[_NBit1, _NBit1]], other: _ArrayLikeComplex_co
    ) -> NDArray[complexfloating[_NBit1, _NBit1]]: ...
    @overload
    def __iadd__(
        self: NDArray[timedelta64], other: _ArrayLikeTD64_co
    ) -> NDArray[timedelta64]: ...
    @overload
    def __iadd__(
        self: NDArray[datetime64], other: _ArrayLikeTD64_co
    ) -> NDArray[datetime64]: ...
    @overload
    def __iadd__(self: NDArray[object_], other: Any) -> NDArray[object_]: ...
    @overload
    def __iadd__(
        self: NDArray[_ScalarType], other: _RecursiveSequence
    ) -> NDArray[_ScalarType]: ...
    @overload
    def __isub__(
        self: NDArray[Any], other: _NestedSequence[Union[str, bytes]]
    ) -> NoReturn: ...
    @overload
    def __isub__(
        self: NDArray[unsignedinteger[_NBit1]], other: _ArrayLikeUInt_co
    ) -> NDArray[unsignedinteger[_NBit1]]: ...
    @overload
    def __isub__(
        self: NDArray[signedinteger[_NBit1]], other: _ArrayLikeInt_co
    ) -> NDArray[signedinteger[_NBit1]]: ...
    @overload
    def __isub__(
        self: NDArray[floating[_NBit1]], other: _ArrayLikeFloat_co
    ) -> NDArray[floating[_NBit1]]: ...
    @overload
    def __isub__(
        self: NDArray[complexfloating[_NBit1, _NBit1]], other: _ArrayLikeComplex_co
    ) -> NDArray[complexfloating[_NBit1, _NBit1]]: ...
    @overload
    def __isub__(
        self: NDArray[timedelta64], other: _ArrayLikeTD64_co
    ) -> NDArray[timedelta64]: ...
    @overload
    def __isub__(
        self: NDArray[datetime64], other: _ArrayLikeTD64_co
    ) -> NDArray[datetime64]: ...
    @overload
    def __isub__(self: NDArray[object_], other: Any) -> NDArray[object_]: ...
    @overload
    def __isub__(
        self: NDArray[_ScalarType], other: _RecursiveSequence
    ) -> NDArray[_ScalarType]: ...
    @overload
    def __imul__(
        self: NDArray[Any], other: _NestedSequence[Union[str, bytes]]
    ) -> NoReturn: ...
    @overload
    def __imul__(self: NDArray[bool_], other: _ArrayLikeBool_co) -> NDArray[bool_]: ...
    @overload
    def __imul__(
        self: NDArray[unsignedinteger[_NBit1]], other: _ArrayLikeUInt_co
    ) -> NDArray[unsignedinteger[_NBit1]]: ...
    @overload
    def __imul__(
        self: NDArray[signedinteger[_NBit1]], other: _ArrayLikeInt_co
    ) -> NDArray[signedinteger[_NBit1]]: ...
    @overload
    def __imul__(
        self: NDArray[floating[_NBit1]], other: _ArrayLikeFloat_co
    ) -> NDArray[floating[_NBit1]]: ...
    @overload
    def __imul__(
        self: NDArray[complexfloating[_NBit1, _NBit1]], other: _ArrayLikeComplex_co
    ) -> NDArray[complexfloating[_NBit1, _NBit1]]: ...
    @overload
    def __imul__(
        self: NDArray[timedelta64], other: _ArrayLikeFloat_co
    ) -> NDArray[timedelta64]: ...
    @overload
    def __imul__(self: NDArray[object_], other: Any) -> NDArray[object_]: ...
    @overload
    def __imul__(
        self: NDArray[_ScalarType], other: _RecursiveSequence
    ) -> NDArray[_ScalarType]: ...
    @overload
    def __itruediv__(
        self: NDArray[Any], other: _NestedSequence[Union[str, bytes]]
    ) -> NoReturn: ...
    @overload
    def __itruediv__(
        self: NDArray[floating[_NBit1]], other: _ArrayLikeFloat_co
    ) -> NDArray[floating[_NBit1]]: ...
    @overload
    def __itruediv__(
        self: NDArray[complexfloating[_NBit1, _NBit1]], other: _ArrayLikeComplex_co
    ) -> NDArray[complexfloating[_NBit1, _NBit1]]: ...
    @overload
    def __itruediv__(
        self: NDArray[timedelta64], other: _ArrayLikeBool_co
    ) -> NoReturn: ...
    @overload
    def __itruediv__(
        self: NDArray[timedelta64], other: _ArrayLikeInt_co
    ) -> NDArray[timedelta64]: ...
    @overload
    def __itruediv__(self: NDArray[object_], other: Any) -> NDArray[object_]: ...
    @overload
    def __itruediv__(
        self: NDArray[_ScalarType], other: _RecursiveSequence
    ) -> NDArray[_ScalarType]: ...
    @overload
    def __ifloordiv__(
        self: NDArray[Any], other: _NestedSequence[Union[str, bytes]]
    ) -> NoReturn: ...
    @overload
    def __ifloordiv__(
        self: NDArray[unsignedinteger[_NBit1]], other: _ArrayLikeUInt_co
    ) -> NDArray[unsignedinteger[_NBit1]]: ...
    @overload
    def __ifloordiv__(
        self: NDArray[signedinteger[_NBit1]], other: _ArrayLikeInt_co
    ) -> NDArray[signedinteger[_NBit1]]: ...
    @overload
    def __ifloordiv__(
        self: NDArray[floating[_NBit1]], other: _ArrayLikeFloat_co
    ) -> NDArray[floating[_NBit1]]: ...
    @overload
    def __ifloordiv__(
        self: NDArray[complexfloating[_NBit1, _NBit1]], other: _ArrayLikeComplex_co
    ) -> NDArray[complexfloating[_NBit1, _NBit1]]: ...
    @overload
    def __ifloordiv__(
        self: NDArray[timedelta64], other: _ArrayLikeBool_co
    ) -> NoReturn: ...
    @overload
    def __ifloordiv__(
        self: NDArray[timedelta64], other: _ArrayLikeInt_co
    ) -> NDArray[timedelta64]: ...
    @overload
    def __ifloordiv__(self: NDArray[object_], other: Any) -> NDArray[object_]: ...
    @overload
    def __ifloordiv__(
        self: NDArray[_ScalarType], other: _RecursiveSequence
    ) -> NDArray[_ScalarType]: ...
    @overload
    def __ipow__(
        self: NDArray[Any], other: _NestedSequence[Union[str, bytes]]
    ) -> NoReturn: ...
    @overload
    def __ipow__(
        self: NDArray[unsignedinteger[_NBit1]], other: _ArrayLikeUInt_co
    ) -> NDArray[unsignedinteger[_NBit1]]: ...
    @overload
    def __ipow__(
        self: NDArray[signedinteger[_NBit1]], other: _ArrayLikeInt_co
    ) -> NDArray[signedinteger[_NBit1]]: ...
    @overload
    def __ipow__(
        self: NDArray[floating[_NBit1]], other: _ArrayLikeFloat_co
    ) -> NDArray[floating[_NBit1]]: ...
    @overload
    def __ipow__(
        self: NDArray[complexfloating[_NBit1, _NBit1]], other: _ArrayLikeComplex_co
    ) -> NDArray[complexfloating[_NBit1, _NBit1]]: ...
    @overload
    def __ipow__(self: NDArray[object_], other: Any) -> NDArray[object_]: ...
    @overload
    def __ipow__(
        self: NDArray[_ScalarType], other: _RecursiveSequence
    ) -> NDArray[_ScalarType]: ...
    @overload
    def __imod__(
        self: NDArray[Any], other: _NestedSequence[Union[str, bytes]]
    ) -> NoReturn: ...
    @overload
    def __imod__(
        self: NDArray[unsignedinteger[_NBit1]], other: _ArrayLikeUInt_co
    ) -> NDArray[unsignedinteger[_NBit1]]: ...
    @overload
    def __imod__(
        self: NDArray[signedinteger[_NBit1]], other: _ArrayLikeInt_co
    ) -> NDArray[signedinteger[_NBit1]]: ...
    @overload
    def __imod__(
        self: NDArray[floating[_NBit1]], other: _ArrayLikeFloat_co
    ) -> NDArray[floating[_NBit1]]: ...
    @overload
    def __imod__(
        self: NDArray[timedelta64],
        other: _NestedSequence[_SupportsArray[_dtype[timedelta64]]],
    ) -> NDArray[timedelta64]: ...
    @overload
    def __imod__(self: NDArray[object_], other: Any) -> NDArray[object_]: ...
    @overload
    def __imod__(
        self: NDArray[_ScalarType], other: _RecursiveSequence
    ) -> NDArray[_ScalarType]: ...
    @overload
    def __ilshift__(
        self: NDArray[Any], other: _NestedSequence[Union[str, bytes]]
    ) -> NoReturn: ...
    @overload
    def __ilshift__(
        self: NDArray[unsignedinteger[_NBit1]], other: _ArrayLikeUInt_co
    ) -> NDArray[unsignedinteger[_NBit1]]: ...
    @overload
    def __ilshift__(
        self: NDArray[signedinteger[_NBit1]], other: _ArrayLikeInt_co
    ) -> NDArray[signedinteger[_NBit1]]: ...
    @overload
    def __ilshift__(self: NDArray[object_], other: Any) -> NDArray[object_]: ...
    @overload
    def __ilshift__(
        self: NDArray[_ScalarType], other: _RecursiveSequence
    ) -> NDArray[_ScalarType]: ...
    @overload
    def __irshift__(
        self: NDArray[Any], other: _NestedSequence[Union[str, bytes]]
    ) -> NoReturn: ...
    @overload
    def __irshift__(
        self: NDArray[unsignedinteger[_NBit1]], other: _ArrayLikeUInt_co
    ) -> NDArray[unsignedinteger[_NBit1]]: ...
    @overload
    def __irshift__(
        self: NDArray[signedinteger[_NBit1]], other: _ArrayLikeInt_co
    ) -> NDArray[signedinteger[_NBit1]]: ...
    @overload
    def __irshift__(self: NDArray[object_], other: Any) -> NDArray[object_]: ...
    @overload
    def __irshift__(
        self: NDArray[_ScalarType], other: _RecursiveSequence
    ) -> NDArray[_ScalarType]: ...
    @overload
    def __iand__(
        self: NDArray[Any], other: _NestedSequence[Union[str, bytes]]
    ) -> NoReturn: ...
    @overload
    def __iand__(self: NDArray[bool_], other: _ArrayLikeBool_co) -> NDArray[bool_]: ...
    @overload
    def __iand__(
        self: NDArray[unsignedinteger[_NBit1]], other: _ArrayLikeUInt_co
    ) -> NDArray[unsignedinteger[_NBit1]]: ...
    @overload
    def __iand__(
        self: NDArray[signedinteger[_NBit1]], other: _ArrayLikeInt_co
    ) -> NDArray[signedinteger[_NBit1]]: ...
    @overload
    def __iand__(self: NDArray[object_], other: Any) -> NDArray[object_]: ...
    @overload
    def __iand__(
        self: NDArray[_ScalarType], other: _RecursiveSequence
    ) -> NDArray[_ScalarType]: ...
    @overload
    def __ixor__(
        self: NDArray[Any], other: _NestedSequence[Union[str, bytes]]
    ) -> NoReturn: ...
    @overload
    def __ixor__(self: NDArray[bool_], other: _ArrayLikeBool_co) -> NDArray[bool_]: ...
    @overload
    def __ixor__(
        self: NDArray[unsignedinteger[_NBit1]], other: _ArrayLikeUInt_co
    ) -> NDArray[unsignedinteger[_NBit1]]: ...
    @overload
    def __ixor__(
        self: NDArray[signedinteger[_NBit1]], other: _ArrayLikeInt_co
    ) -> NDArray[signedinteger[_NBit1]]: ...
    @overload
    def __ixor__(self: NDArray[object_], other: Any) -> NDArray[object_]: ...
    @overload
    def __ixor__(
        self: NDArray[_ScalarType], other: _RecursiveSequence
    ) -> NDArray[_ScalarType]: ...
    @overload
    def __ior__(
        self: NDArray[Any], other: _NestedSequence[Union[str, bytes]]
    ) -> NoReturn: ...
    @overload
    def __ior__(self: NDArray[bool_], other: _ArrayLikeBool_co) -> NDArray[bool_]: ...
    @overload
    def __ior__(
        self: NDArray[unsignedinteger[_NBit1]], other: _ArrayLikeUInt_co
    ) -> NDArray[unsignedinteger[_NBit1]]: ...
    @overload
    def __ior__(
        self: NDArray[signedinteger[_NBit1]], other: _ArrayLikeInt_co
    ) -> NDArray[signedinteger[_NBit1]]: ...
    @overload
    def __ior__(self: NDArray[object_], other: Any) -> NDArray[object_]: ...
    @overload
    def __ior__(
        self: NDArray[_ScalarType], other: _RecursiveSequence
    ) -> NDArray[_ScalarType]: ...
    @property
    def dtype(self) -> _DType_co: ...

_ScalarType = ...
_NBit1 = ...
_NBit2 = ...

class generic(_ArrayOrScalarCommon):
    @abstractmethod
    def __init__(self, *args: Any, **kwargs: Any) -> None: ...
    @overload
    def __array__(
        self: _ScalarType, __dtype: None = ...
    ) -> ndarray[Any, _dtype[_ScalarType]]: ...
    @overload
    def __array__(self, __dtype: _DType) -> ndarray[Any, _DType]: ...
    @property
    def base(self) -> None: ...
    @property
    def ndim(self) -> L[0]: ...
    @property
    def size(self) -> L[1]: ...
    @property
    def shape(self) -> Tuple[()]: ...
    @property
    def strides(self) -> Tuple[()]: ...
    def byteswap(self: _ScalarType, inplace: L[False] = ...) -> _ScalarType: ...
    @property
    def flat(self: _ScalarType) -> flatiter[ndarray[Any, _dtype[_ScalarType]]]: ...
    @overload
    def astype(
        self,
        dtype: _DTypeLike[_ScalarType],
        order: _OrderKACF = ...,
        casting: _Casting = ...,
        subok: bool = ...,
        copy: bool = ...,
    ) -> _ScalarType: ...
    @overload
    def astype(
        self,
        dtype: DTypeLike,
        order: _OrderKACF = ...,
        casting: _Casting = ...,
        subok: bool = ...,
        copy: bool = ...,
    ) -> Any: ...
    @overload
    def view(self: _ScalarType, type: Type[ndarray[Any, Any]] = ...) -> _ScalarType: ...
    @overload
    def view(
        self, dtype: _DTypeLike[_ScalarType], type: Type[ndarray[Any, Any]] = ...
    ) -> _ScalarType: ...
    @overload
    def view(self, dtype: DTypeLike, type: Type[ndarray[Any, Any]] = ...) -> Any: ...
    @overload
    def getfield(
        self, dtype: _DTypeLike[_ScalarType], offset: SupportsIndex = ...
    ) -> _ScalarType: ...
    @overload
    def getfield(self, dtype: DTypeLike, offset: SupportsIndex = ...) -> Any: ...
    def item(self, __args: Union[L[0], Tuple[()], Tuple[L[0]]] = ...) -> Any: ...
    @overload
    def take(
        self: _ScalarType,
        indices: _IntLike_co,
        axis: Optional[SupportsIndex] = ...,
        out: None = ...,
        mode: _ModeKind = ...,
    ) -> _ScalarType: ...
    @overload
    def take(
        self: _ScalarType,
        indices: _ArrayLikeInt_co,
        axis: Optional[SupportsIndex] = ...,
        out: None = ...,
        mode: _ModeKind = ...,
    ) -> ndarray[Any, _dtype[_ScalarType]]: ...
    @overload
    def take(
        self,
        indices: _ArrayLikeInt_co,
        axis: Optional[SupportsIndex] = ...,
        out: _NdArraySubClass = ...,
        mode: _ModeKind = ...,
    ) -> _NdArraySubClass: ...
    def repeat(
        self: _ScalarType, repeats: _ArrayLikeInt_co, axis: Optional[SupportsIndex] = ...
    ) -> ndarray[Any, _dtype[_ScalarType]]: ...
    def flatten(
        self: _ScalarType, order: _OrderKACF = ...
    ) -> ndarray[Any, _dtype[_ScalarType]]: ...
    def ravel(
        self: _ScalarType, order: _OrderKACF = ...
    ) -> ndarray[Any, _dtype[_ScalarType]]: ...
    @overload
    def reshape(
        self: _ScalarType, __shape: _ShapeLike, *, order: _OrderACF = ...
    ) -> ndarray[Any, _dtype[_ScalarType]]: ...
    @overload
    def reshape(
        self: _ScalarType, *shape: SupportsIndex, order: _OrderACF = ...
    ) -> ndarray[Any, _dtype[_ScalarType]]: ...
    def squeeze(self: _ScalarType, axis: Union[L[0], Tuple[()]] = ...) -> _ScalarType: ...
    def transpose(self: _ScalarType, __axes: Tuple[()] = ...) -> _ScalarType: ...
    @property
    def dtype(self: _ScalarType) -> _dtype[_ScalarType]: ...

class number(generic, Generic[_NBit1]):
    @property
    def real(self: _ArraySelf) -> _ArraySelf: ...
    @property
    def imag(self: _ArraySelf) -> _ArraySelf: ...
    def __int__(self) -> int: ...
    def __float__(self) -> float: ...
    def __complex__(self) -> complex: ...
    def __neg__(self: _ArraySelf) -> _ArraySelf: ...
    def __pos__(self: _ArraySelf) -> _ArraySelf: ...
    def __abs__(self: _ArraySelf) -> _ArraySelf: ...
    __add__: _NumberOp
    __radd__: _NumberOp
    __sub__: _NumberOp
    __rsub__: _NumberOp
    __mul__: _NumberOp
    __rmul__: _NumberOp
    __floordiv__: _NumberOp
    __rfloordiv__: _NumberOp
    __pow__: _NumberOp
    __rpow__: _NumberOp
    __truediv__: _NumberOp
    __rtruediv__: _NumberOp
    __lt__: _ComparisonOp[_NumberLike_co, _ArrayLikeNumber_co]
    __le__: _ComparisonOp[_NumberLike_co, _ArrayLikeNumber_co]
    __gt__: _ComparisonOp[_NumberLike_co, _ArrayLikeNumber_co]
    __ge__: _ComparisonOp[_NumberLike_co, _ArrayLikeNumber_co]

class bool_(generic):
    def __init__(self, __value: object = ...) -> None: ...
    def item(self, __args: Union[L[0], Tuple[()], Tuple[L[0]]] = ...) -> bool: ...
    def tolist(self) -> bool: ...
    @property
    def real(self: _ArraySelf) -> _ArraySelf: ...
    @property
    def imag(self: _ArraySelf) -> _ArraySelf: ...
    def __int__(self) -> int: ...
    def __float__(self) -> float: ...
    def __complex__(self) -> complex: ...
    def __abs__(self: _ArraySelf) -> _ArraySelf: ...
    __add__: _BoolOp[bool_]
    __radd__: _BoolOp[bool_]
    __sub__: _BoolSub
    __rsub__: _BoolSub
    __mul__: _BoolOp[bool_]
    __rmul__: _BoolOp[bool_]
    __floordiv__: _BoolOp[int8]
    __rfloordiv__: _BoolOp[int8]
    __pow__: _BoolOp[int8]
    __rpow__: _BoolOp[int8]
    __truediv__: _BoolTrueDiv
    __rtruediv__: _BoolTrueDiv
    def __invert__(self) -> bool_: ...
    __lshift__: _BoolBitOp[int8]
    __rlshift__: _BoolBitOp[int8]
    __rshift__: _BoolBitOp[int8]
    __rrshift__: _BoolBitOp[int8]
    __and__: _BoolBitOp[bool_]
    __rand__: _BoolBitOp[bool_]
    __xor__: _BoolBitOp[bool_]
    __rxor__: _BoolBitOp[bool_]
    __or__: _BoolBitOp[bool_]
    __ror__: _BoolBitOp[bool_]
    __mod__: _BoolMod
    __rmod__: _BoolMod
    __divmod__: _BoolDivMod
    __rdivmod__: _BoolDivMod
    __lt__: _ComparisonOp[_NumberLike_co, _ArrayLikeNumber_co]
    __le__: _ComparisonOp[_NumberLike_co, _ArrayLikeNumber_co]
    __gt__: _ComparisonOp[_NumberLike_co, _ArrayLikeNumber_co]
    __ge__: _ComparisonOp[_NumberLike_co, _ArrayLikeNumber_co]

bool8 = bool_

class object_(generic):
    def __init__(self, __value: object = ...) -> None: ...
    @property
    def real(self: _ArraySelf) -> _ArraySelf: ...
    @property
    def imag(self: _ArraySelf) -> _ArraySelf: ...
    def __int__(self) -> int: ...
    def __float__(self) -> float: ...
    def __complex__(self) -> complex: ...

object0 = object_

class _DatetimeScalar(Protocol):
    @property
    def day(self) -> int: ...
    @property
    def month(self) -> int: ...
    @property
    def year(self) -> int: ...

class datetime64(generic):
    @overload
    def __init__(
        self,
        __value: Union[None, datetime64, _CharLike_co, _DatetimeScalar] = ...,
        __format: Union[_CharLike_co, Tuple[_CharLike_co, _IntLike_co]] = ...,
    ) -> None: ...
    @overload
    def __init__(
        self,
        __value: int,
        __format: Union[_CharLike_co, Tuple[_CharLike_co, _IntLike_co]],
    ) -> None: ...
    def __add__(self, other: _TD64Like_co) -> datetime64: ...
    def __radd__(self, other: _TD64Like_co) -> datetime64: ...
    @overload
    def __sub__(self, other: datetime64) -> timedelta64: ...
    @overload
    def __sub__(self, other: _TD64Like_co) -> datetime64: ...
    def __rsub__(self, other: datetime64) -> timedelta64: ...
    __lt__: _ComparisonOp[datetime64, _ArrayLikeDT64_co]
    __le__: _ComparisonOp[datetime64, _ArrayLikeDT64_co]
    __gt__: _ComparisonOp[datetime64, _ArrayLikeDT64_co]
    __ge__: _ComparisonOp[datetime64, _ArrayLikeDT64_co]

if sys.version_info >= (3, 8):
    _IntValue = Union[SupportsInt, _CharLike_co, SupportsIndex]
    _FloatValue = Union[None, _CharLike_co, SupportsFloat, SupportsIndex]
    _ComplexValue = (
        Union[None, _CharLike_co, SupportsFloat, SupportsComplex, SupportsIndex, complex],
    )
else: ...

class integer(number[_NBit1]):
    @property
    def numerator(self: _ScalarType) -> _ScalarType: ...
    @property
    def denominator(self) -> L[1]: ...
    @overload
    def __round__(self, ndigits: None = ...) -> int: ...
    @overload
    def __round__(self: _ScalarType, ndigits: SupportsIndex) -> _ScalarType: ...
    def item(self, __args: Union[L[0], Tuple[()], Tuple[L[0]]] = ...) -> int: ...
    def tolist(self) -> int: ...
    def __index__(self) -> int: ...
    __truediv__: _IntTrueDiv[_NBit1]
    __rtruediv__: _IntTrueDiv[_NBit1]
    def __mod__(self, value: _IntLike_co) -> integer: ...
    def __rmod__(self, value: _IntLike_co) -> integer: ...
    def __invert__(self: _IntType) -> _IntType: ...
    def __lshift__(self, other: _IntLike_co) -> integer: ...
    def __rlshift__(self, other: _IntLike_co) -> integer: ...
    def __rshift__(self, other: _IntLike_co) -> integer: ...
    def __rrshift__(self, other: _IntLike_co) -> integer: ...
    def __and__(self, other: _IntLike_co) -> integer: ...
    def __rand__(self, other: _IntLike_co) -> integer: ...
    def __or__(self, other: _IntLike_co) -> integer: ...
    def __ror__(self, other: _IntLike_co) -> integer: ...
    def __xor__(self, other: _IntLike_co) -> integer: ...
    def __rxor__(self, other: _IntLike_co) -> integer: ...

class signedinteger(integer[_NBit1]):
    def __init__(self, __value: _IntValue = ...) -> None: ...
    __add__: _SignedIntOp[_NBit1]
    __radd__: _SignedIntOp[_NBit1]
    __sub__: _SignedIntOp[_NBit1]
    __rsub__: _SignedIntOp[_NBit1]
    __mul__: _SignedIntOp[_NBit1]
    __rmul__: _SignedIntOp[_NBit1]
    __floordiv__: _SignedIntOp[_NBit1]
    __rfloordiv__: _SignedIntOp[_NBit1]
    __pow__: _SignedIntOp[_NBit1]
    __rpow__: _SignedIntOp[_NBit1]
    __lshift__: _SignedIntBitOp[_NBit1]
    __rlshift__: _SignedIntBitOp[_NBit1]
    __rshift__: _SignedIntBitOp[_NBit1]
    __rrshift__: _SignedIntBitOp[_NBit1]
    __and__: _SignedIntBitOp[_NBit1]
    __rand__: _SignedIntBitOp[_NBit1]
    __xor__: _SignedIntBitOp[_NBit1]
    __rxor__: _SignedIntBitOp[_NBit1]
    __or__: _SignedIntBitOp[_NBit1]
    __ror__: _SignedIntBitOp[_NBit1]
    __mod__: _SignedIntMod[_NBit1]
    __rmod__: _SignedIntMod[_NBit1]
    __divmod__: _SignedIntDivMod[_NBit1]
    __rdivmod__: _SignedIntDivMod[_NBit1]

int8 = signedinteger[_8Bit]
int16 = signedinteger[_16Bit]
int32 = signedinteger[_32Bit]
int64 = signedinteger[_64Bit]
byte = signedinteger[_NBitByte]
short = signedinteger[_NBitShort]
intc = signedinteger[_NBitIntC]
intp = signedinteger[_NBitIntP]
int0 = signedinteger[_NBitIntP]
int_ = signedinteger[_NBitInt]
longlong = signedinteger[_NBitLongLong]

class timedelta64(generic):
    def __init__(
        self,
        __value: Union[None, int, _CharLike_co, dt.timedelta, timedelta64] = ...,
        __format: Union[_CharLike_co, Tuple[_CharLike_co, _IntLike_co]] = ...,
    ) -> None: ...
    @property
    def numerator(self: _ScalarType) -> _ScalarType: ...
    @property
    def denominator(self) -> L[1]: ...
    def __int__(self) -> int: ...
    def __float__(self) -> float: ...
    def __complex__(self) -> complex: ...
    def __neg__(self: _ArraySelf) -> _ArraySelf: ...
    def __pos__(self: _ArraySelf) -> _ArraySelf: ...
    def __abs__(self: _ArraySelf) -> _ArraySelf: ...
    def __add__(self, other: _TD64Like_co) -> timedelta64: ...
    def __radd__(self, other: _TD64Like_co) -> timedelta64: ...
    def __sub__(self, other: _TD64Like_co) -> timedelta64: ...
    def __rsub__(self, other: _TD64Like_co) -> timedelta64: ...
    def __mul__(self, other: _FloatLike_co) -> timedelta64: ...
    def __rmul__(self, other: _FloatLike_co) -> timedelta64: ...
    __truediv__: _TD64Div[float64]
    __floordiv__: _TD64Div[int64]
    def __rtruediv__(self, other: timedelta64) -> float64: ...
    def __rfloordiv__(self, other: timedelta64) -> int64: ...
    def __mod__(self, other: timedelta64) -> timedelta64: ...
    def __rmod__(self, other: timedelta64) -> timedelta64: ...
    def __divmod__(self, other: timedelta64) -> Tuple[int64, timedelta64]: ...
    def __rdivmod__(self, other: timedelta64) -> Tuple[int64, timedelta64]: ...
    __lt__: _ComparisonOp[_TD64Like_co, _ArrayLikeTD64_co]
    __le__: _ComparisonOp[_TD64Like_co, _ArrayLikeTD64_co]
    __gt__: _ComparisonOp[_TD64Like_co, _ArrayLikeTD64_co]
    __ge__: _ComparisonOp[_TD64Like_co, _ArrayLikeTD64_co]

class unsignedinteger(integer[_NBit1]):
    def __init__(self, __value: _IntValue = ...) -> None: ...
    __add__: _UnsignedIntOp[_NBit1]
    __radd__: _UnsignedIntOp[_NBit1]
    __sub__: _UnsignedIntOp[_NBit1]
    __rsub__: _UnsignedIntOp[_NBit1]
    __mul__: _UnsignedIntOp[_NBit1]
    __rmul__: _UnsignedIntOp[_NBit1]
    __floordiv__: _UnsignedIntOp[_NBit1]
    __rfloordiv__: _UnsignedIntOp[_NBit1]
    __pow__: _UnsignedIntOp[_NBit1]
    __rpow__: _UnsignedIntOp[_NBit1]
    __lshift__: _UnsignedIntBitOp[_NBit1]
    __rlshift__: _UnsignedIntBitOp[_NBit1]
    __rshift__: _UnsignedIntBitOp[_NBit1]
    __rrshift__: _UnsignedIntBitOp[_NBit1]
    __and__: _UnsignedIntBitOp[_NBit1]
    __rand__: _UnsignedIntBitOp[_NBit1]
    __xor__: _UnsignedIntBitOp[_NBit1]
    __rxor__: _UnsignedIntBitOp[_NBit1]
    __or__: _UnsignedIntBitOp[_NBit1]
    __ror__: _UnsignedIntBitOp[_NBit1]
    __mod__: _UnsignedIntMod[_NBit1]
    __rmod__: _UnsignedIntMod[_NBit1]
    __divmod__: _UnsignedIntDivMod[_NBit1]
    __rdivmod__: _UnsignedIntDivMod[_NBit1]

uint8 = unsignedinteger[_8Bit]
uint16 = unsignedinteger[_16Bit]
uint32 = unsignedinteger[_32Bit]
uint64 = unsignedinteger[_64Bit]
ubyte = unsignedinteger[_NBitByte]
ushort = unsignedinteger[_NBitShort]
uintc = unsignedinteger[_NBitIntC]
uintp = unsignedinteger[_NBitIntP]
uint0 = unsignedinteger[_NBitIntP]
uint = unsignedinteger[_NBitInt]
ulonglong = unsignedinteger[_NBitLongLong]

class inexact(number[_NBit1]):
    def __getnewargs__(self: inexact[_64Bit]) -> Tuple[float, ...]: ...

_IntType = ...
_FloatType = ...

class floating(inexact[_NBit1]):
    def __init__(self, __value: _FloatValue = ...) -> None: ...
    def item(self, __args: Union[L[0], Tuple[()], Tuple[L[0]]] = ...) -> float: ...
    def tolist(self) -> float: ...
    def is_integer(self: float64) -> bool: ...
    def hex(self: float64) -> str: ...
    @classmethod
    def fromhex(cls: Type[float64], __string: str) -> float64: ...
    def as_integer_ratio(self) -> Tuple[int, int]: ...
    if sys.version_info >= (3, 9): ...
    def __trunc__(self: float64) -> int: ...
    def __getnewargs__(self: float64) -> Tuple[float]: ...
    def __getformat__(self: float64, __typestr: L["double", "float"]) -> str: ...
    @overload
    def __round__(self, ndigits: None = ...) -> int: ...
    @overload
    def __round__(self: _ScalarType, ndigits: SupportsIndex) -> _ScalarType: ...
    __add__: _FloatOp[_NBit1]
    __radd__: _FloatOp[_NBit1]
    __sub__: _FloatOp[_NBit1]
    __rsub__: _FloatOp[_NBit1]
    __mul__: _FloatOp[_NBit1]
    __rmul__: _FloatOp[_NBit1]
    __truediv__: _FloatOp[_NBit1]
    __rtruediv__: _FloatOp[_NBit1]
    __floordiv__: _FloatOp[_NBit1]
    __rfloordiv__: _FloatOp[_NBit1]
    __pow__: _FloatOp[_NBit1]
    __rpow__: _FloatOp[_NBit1]
    __mod__: _FloatMod[_NBit1]
    __rmod__: _FloatMod[_NBit1]
    __divmod__: _FloatDivMod[_NBit1]
    __rdivmod__: _FloatDivMod[_NBit1]

float16 = floating[_16Bit]
float32 = floating[_32Bit]
float64 = floating[_64Bit]
half = floating[_NBitHalf]
single = floating[_NBitSingle]
double = floating[_NBitDouble]
float_ = floating[_NBitDouble]
longdouble = floating[_NBitLongDouble]
longfloat = floating[_NBitLongDouble]

class complexfloating(inexact[_NBit1], Generic[_NBit1, _NBit2]):
    def __init__(self, __value: _ComplexValue = ...) -> None: ...
    def item(self, __args: Union[L[0], Tuple[()], Tuple[L[0]]] = ...) -> complex: ...
    def tolist(self) -> complex: ...
    @property
    def real(self) -> floating[_NBit1]: ...
    @property
    def imag(self) -> floating[_NBit2]: ...
    def __abs__(self) -> floating[_NBit1]: ...
    def __getnewargs__(self: complex128) -> Tuple[float, float]: ...
    __add__: _ComplexOp[_NBit1]
    __radd__: _ComplexOp[_NBit1]
    __sub__: _ComplexOp[_NBit1]
    __rsub__: _ComplexOp[_NBit1]
    __mul__: _ComplexOp[_NBit1]
    __rmul__: _ComplexOp[_NBit1]
    __truediv__: _ComplexOp[_NBit1]
    __rtruediv__: _ComplexOp[_NBit1]
    __floordiv__: _ComplexOp[_NBit1]
    __rfloordiv__: _ComplexOp[_NBit1]
    __pow__: _ComplexOp[_NBit1]
    __rpow__: _ComplexOp[_NBit1]

complex64 = complexfloating[_32Bit, _32Bit]
complex128 = complexfloating[_64Bit, _64Bit]
csingle = complexfloating[_NBitSingle, _NBitSingle]
singlecomplex = complexfloating[_NBitSingle, _NBitSingle]
cdouble = complexfloating[_NBitDouble, _NBitDouble]
complex_ = complexfloating[_NBitDouble, _NBitDouble]
cfloat = complexfloating[_NBitDouble, _NBitDouble]
clongdouble = complexfloating[_NBitLongDouble, _NBitLongDouble]
clongfloat = complexfloating[_NBitLongDouble, _NBitLongDouble]
longcomplex = complexfloating[_NBitLongDouble, _NBitLongDouble]

class flexible(generic): ...

class void(flexible):
    def __init__(self, __value: Union[_IntLike_co, bytes]) -> None: ...
    @property
    def real(self: _ArraySelf) -> _ArraySelf: ...
    @property
    def imag(self: _ArraySelf) -> _ArraySelf: ...
    def setfield(self, val: ArrayLike, dtype: DTypeLike, offset: int = ...) -> None: ...
    def __getitem__(self, key: SupportsIndex) -> Any: ...
    def __setitem__(self, key: SupportsIndex, value: ArrayLike) -> None: ...

void0 = void

class character(flexible):
    def __int__(self) -> int: ...
    def __float__(self) -> float: ...

class bytes_(character, bytes):
    @overload
    def __init__(self, __value: object = ...) -> None: ...
    @overload
    def __init__(self, __value: str, encoding: str = ..., errors: str = ...) -> None: ...
    def item(self, __args: Union[L[0], Tuple[()], Tuple[L[0]]] = ...) -> bytes: ...
    def tolist(self) -> bytes: ...

string_ = bytes_
bytes0 = bytes_

class str_(character, str):
    @overload
    def __init__(self, __value: object = ...) -> None: ...
    @overload
    def __init__(
        self, __value: bytes, encoding: str = ..., errors: str = ...
    ) -> None: ...
    def item(self, __args: Union[L[0], Tuple[()], Tuple[L[0]]] = ...) -> str: ...
    def tolist(self) -> str: ...

unicode_ = str_
str0 = str_

def array(
    object: object,
    dtype: DTypeLike = ...,
    *,
    copy: bool = ...,
    order: _OrderKACF = ...,
    subok: bool = ...,
    ndmin: int = ...,
    like: ArrayLike = ...
) -> ndarray: ...
def zeros(
    shape: _ShapeLike,
    dtype: DTypeLike = ...,
    order: _OrderCF = ...,
    *,
    like: ArrayLike = ...
) -> ndarray: ...
def empty(
    shape: _ShapeLike,
    dtype: DTypeLike = ...,
    order: _OrderCF = ...,
    *,
    like: ArrayLike = ...
) -> ndarray: ...

Inf: Final[float]
Infinity: Final[float]
NAN: Final[float]
NINF: Final[float]
NZERO: Final[float]
NaN: Final[float]
PINF: Final[float]
PZERO: Final[float]
e: Final[float]
euler_gamma: Final[float]
inf: Final[float]
infty: Final[float]
nan: Final[float]
pi: Final[float]
ALLOW_THREADS: Final[int]
BUFSIZE: Final[int]
CLIP: Final[int]
ERR_CALL: Final[int]
ERR_DEFAULT: Final[int]
ERR_IGNORE: Final[int]
ERR_LOG: Final[int]
ERR_PRINT: Final[int]
ERR_RAISE: Final[int]
ERR_WARN: Final[int]
FLOATING_POINT_SUPPORT: Final[int]
FPE_DIVIDEBYZERO: Final[int]
FPE_INVALID: Final[int]
FPE_OVERFLOW: Final[int]
FPE_UNDERFLOW: Final[int]
MAXDIMS: Final[int]
MAY_SHARE_BOUNDS: Final[int]
MAY_SHARE_EXACT: Final[int]
RAISE: Final[int]
SHIFT_DIVIDEBYZERO: Final[int]
SHIFT_INVALID: Final[int]
SHIFT_OVERFLOW: Final[int]
SHIFT_UNDERFLOW: Final[int]
UFUNC_BUFSIZE_DEFAULT: Final[int]
WRAP: Final[int]
tracemalloc_domain: Final[int]
little_endian: Final[bool]
True_: Final[bool_]
False_: Final[bool_]
UFUNC_PYVALS_NAME: Final[str]
newaxis: None

class ufunc:
    @property
    def __name__(self) -> str: ...
    @property
    def __doc__(self) -> str: ...
    __call__: Callable[..., Any]
    @property
    def nin(self) -> int: ...
    @property
    def nout(self) -> int: ...
    @property
    def nargs(self) -> int: ...
    @property
    def ntypes(self) -> int: ...
    @property
    def types(self) -> List[str]: ...
    @property
    def identity(self) -> Any: ...
    @property
    def signature(self) -> Optional[str]: ...
    reduce: Any
    accumulate: Any
    reduce: Any
    outer: Any
    at: Any

absolute: _UFunc_Nin1_Nout1[L["absolute"], L[20], None]
add: _UFunc_Nin2_Nout1[L["add"], L[22], L[0]]
arccos: _UFunc_Nin1_Nout1[L["arccos"], L[8], None]
arccosh: _UFunc_Nin1_Nout1[L["arccosh"], L[8], None]
arcsin: _UFunc_Nin1_Nout1[L["arcsin"], L[8], None]
arcsinh: _UFunc_Nin1_Nout1[L["arcsinh"], L[8], None]
arctan2: _UFunc_Nin2_Nout1[L["arctan2"], L[5], None]
arctan: _UFunc_Nin1_Nout1[L["arctan"], L[8], None]
arctanh: _UFunc_Nin1_Nout1[L["arctanh"], L[8], None]
bitwise_and: _UFunc_Nin2_Nout1[L["bitwise_and"], L[12], L[-1]]
bitwise_not: _UFunc_Nin1_Nout1[L["invert"], L[12], None]
bitwise_or: _UFunc_Nin2_Nout1[L["bitwise_or"], L[12], L[0]]
bitwise_xor: _UFunc_Nin2_Nout1[L["bitwise_xor"], L[12], L[0]]
cbrt: _UFunc_Nin1_Nout1[L["cbrt"], L[5], None]
ceil: _UFunc_Nin1_Nout1[L["ceil"], L[7], None]
conj: _UFunc_Nin1_Nout1[L["conjugate"], L[18], None]
conjugate: _UFunc_Nin1_Nout1[L["conjugate"], L[18], None]
copysign: _UFunc_Nin2_Nout1[L["copysign"], L[4], None]
cos: _UFunc_Nin1_Nout1[L["cos"], L[9], None]
cosh: _UFunc_Nin1_Nout1[L["cosh"], L[8], None]
deg2rad: _UFunc_Nin1_Nout1[L["deg2rad"], L[5], None]
degrees: _UFunc_Nin1_Nout1[L["degrees"], L[5], None]
divide: _UFunc_Nin2_Nout1[L["true_divide"], L[11], None]
divmod: _UFunc_Nin2_Nout2[L["divmod"], L[15], None]
equal: _UFunc_Nin2_Nout1[L["equal"], L[23], None]
exp2: _UFunc_Nin1_Nout1[L["exp2"], L[8], None]
exp: _UFunc_Nin1_Nout1[L["exp"], L[10], None]
expm1: _UFunc_Nin1_Nout1[L["expm1"], L[8], None]
fabs: _UFunc_Nin1_Nout1[L["fabs"], L[5], None]
float_power: _UFunc_Nin2_Nout1[L["float_power"], L[4], None]
floor: _UFunc_Nin1_Nout1[L["floor"], L[7], None]
floor_divide: _UFunc_Nin2_Nout1[L["floor_divide"], L[21], None]
fmax: _UFunc_Nin2_Nout1[L["fmax"], L[21], None]
fmin: _UFunc_Nin2_Nout1[L["fmin"], L[21], None]
fmod: _UFunc_Nin2_Nout1[L["fmod"], L[15], None]
frexp: _UFunc_Nin1_Nout2[L["frexp"], L[4], None]
gcd: _UFunc_Nin2_Nout1[L["gcd"], L[11], L[0]]
greater: _UFunc_Nin2_Nout1[L["greater"], L[23], None]
greater_equal: _UFunc_Nin2_Nout1[L["greater_equal"], L[23], None]
heaviside: _UFunc_Nin2_Nout1[L["heaviside"], L[4], None]
hypot: _UFunc_Nin2_Nout1[L["hypot"], L[5], L[0]]
invert: _UFunc_Nin1_Nout1[L["invert"], L[12], None]
isfinite: _UFunc_Nin1_Nout1[L["isfinite"], L[20], None]
isinf: _UFunc_Nin1_Nout1[L["isinf"], L[20], None]
isnan: _UFunc_Nin1_Nout1[L["isnan"], L[20], None]
isnat: _UFunc_Nin1_Nout1[L["isnat"], L[2], None]
lcm: _UFunc_Nin2_Nout1[L["lcm"], L[11], None]
ldexp: _UFunc_Nin2_Nout1[L["ldexp"], L[8], None]
left_shift: _UFunc_Nin2_Nout1[L["left_shift"], L[11], None]
less: _UFunc_Nin2_Nout1[L["less"], L[23], None]
less_equal: _UFunc_Nin2_Nout1[L["less_equal"], L[23], None]
log10: _UFunc_Nin1_Nout1[L["log10"], L[8], None]
log1p: _UFunc_Nin1_Nout1[L["log1p"], L[8], None]
log2: _UFunc_Nin1_Nout1[L["log2"], L[8], None]
log: _UFunc_Nin1_Nout1[L["log"], L[10], None]
logaddexp2: _UFunc_Nin2_Nout1[L["logaddexp2"], L[4], float]
logaddexp: _UFunc_Nin2_Nout1[L["logaddexp"], L[4], float]
logical_and: _UFunc_Nin2_Nout1[L["logical_and"], L[20], L[True]]
logical_not: _UFunc_Nin1_Nout1[L["logical_not"], L[20], None]
logical_or: _UFunc_Nin2_Nout1[L["logical_or"], L[20], L[False]]
logical_xor: _UFunc_Nin2_Nout1[L["logical_xor"], L[19], L[False]]
matmul: _GUFunc_Nin2_Nout1[L["matmul"], L[19], None]
maximum: _UFunc_Nin2_Nout1[L["maximum"], L[21], None]
minimum: _UFunc_Nin2_Nout1[L["minimum"], L[21], None]
mod: _UFunc_Nin2_Nout1[L["remainder"], L[16], None]
modf: _UFunc_Nin1_Nout2[L["modf"], L[4], None]
multiply: _UFunc_Nin2_Nout1[L["multiply"], L[23], L[1]]
negative: _UFunc_Nin1_Nout1[L["negative"], L[19], None]
nextafter: _UFunc_Nin2_Nout1[L["nextafter"], L[4], None]
not_equal: _UFunc_Nin2_Nout1[L["not_equal"], L[23], None]
positive: _UFunc_Nin1_Nout1[L["positive"], L[19], None]
power: _UFunc_Nin2_Nout1[L["power"], L[18], None]
rad2deg: _UFunc_Nin1_Nout1[L["rad2deg"], L[5], None]
radians: _UFunc_Nin1_Nout1[L["radians"], L[5], None]
reciprocal: _UFunc_Nin1_Nout1[L["reciprocal"], L[18], None]
remainder: _UFunc_Nin2_Nout1[L["remainder"], L[16], None]
right_shift: _UFunc_Nin2_Nout1[L["right_shift"], L[11], None]
rint: _UFunc_Nin1_Nout1[L["rint"], L[10], None]
sign: _UFunc_Nin1_Nout1[L["sign"], L[19], None]
signbit: _UFunc_Nin1_Nout1[L["signbit"], L[4], None]
sin: _UFunc_Nin1_Nout1[L["sin"], L[9], None]
sinh: _UFunc_Nin1_Nout1[L["sinh"], L[8], None]
spacing: _UFunc_Nin1_Nout1[L["spacing"], L[4], None]
sqrt: _UFunc_Nin1_Nout1[L["sqrt"], L[10], None]
square: _UFunc_Nin1_Nout1[L["square"], L[18], None]
subtract: _UFunc_Nin2_Nout1[L["subtract"], L[21], None]
tan: _UFunc_Nin1_Nout1[L["tan"], L[8], None]
tanh: _UFunc_Nin1_Nout1[L["tanh"], L[8], None]
true_divide: _UFunc_Nin2_Nout1[L["true_divide"], L[11], None]
trunc: _UFunc_Nin1_Nout1[L["trunc"], L[7], None]
abs = ...

class ModuleDeprecationWarning(DeprecationWarning): ...
class VisibleDeprecationWarning(UserWarning): ...
class ComplexWarning(RuntimeWarning): ...
class RankWarning(UserWarning): ...
class TooHardError(RuntimeError): ...

class AxisError(ValueError, IndexError):
    def __init__(
        self, axis: int, ndim: Optional[int] = ..., msg_prefix: Optional[str] = ...
    ) -> None: ...

_CallType = ...

class errstate(Generic[_CallType], ContextDecorator):
    call: _CallType
    kwargs: _ErrDictOptional
    def __init__(
        self,
        *,
        call: _CallType = ...,
        all: Optional[_ErrKind] = ...,
        divide: Optional[_ErrKind] = ...,
        over: Optional[_ErrKind] = ...,
        under: Optional[_ErrKind] = ...,
        invalid: Optional[_ErrKind] = ...
    ) -> None: ...
    def __enter__(self) -> None: ...
    def __exit__(
        self,
        __exc_type: Optional[Type[BaseException]],
        __exc_value: Optional[BaseException],
        __traceback: Optional[TracebackType],
    ) -> None: ...

class ndenumerate(Generic[_ScalarType]):
    iter: flatiter[NDArray[_ScalarType]]
    @overload
    def __new__(
        cls, arr: _NestedSequence[_SupportsArray[dtype[_ScalarType]]]
    ) -> ndenumerate[_ScalarType]: ...
    @overload
    def __new__(cls, arr: _NestedSequence[str]) -> ndenumerate[str_]: ...
    @overload
    def __new__(cls, arr: _NestedSequence[bytes]) -> ndenumerate[bytes_]: ...
    @overload
    def __new__(cls, arr: _NestedSequence[bool]) -> ndenumerate[bool_]: ...
    @overload
    def __new__(cls, arr: _NestedSequence[int]) -> ndenumerate[int_]: ...
    @overload
    def __new__(cls, arr: _NestedSequence[float]) -> ndenumerate[float_]: ...
    @overload
    def __new__(cls, arr: _NestedSequence[complex]) -> ndenumerate[complex_]: ...
    @overload
    def __new__(cls, arr: _RecursiveSequence) -> ndenumerate[Any]: ...
    def __next__(self: ndenumerate[_ScalarType]) -> Tuple[_Shape, _ScalarType]: ...
    def __iter__(self: _T) -> _T: ...

class ndindex:
    def __init__(self, *shape: SupportsIndex) -> None: ...
    def __iter__(self: _T) -> _T: ...
    def __next__(self) -> _Shape: ...

class DataSource:
    def __init__(self, destpath: Union[None, str, os.PathLike[str]] = ...) -> None: ...
    def __del__(self) -> None: ...
    def abspath(self, path: str) -> str: ...
    def exists(self, path: str) -> bool: ...
    def open(
        self,
        path: str,
        mode: str = ...,
        encoding: Optional[str] = ...,
        newline: Optional[str] = ...,
    ) -> IO[Any]: ...

class broadcast:
    def __new__(cls, *args: ArrayLike) -> broadcast: ...
    @property
    def index(self) -> int: ...
    @property
    def iters(self) -> Tuple[flatiter[Any], ...]: ...
    @property
    def nd(self) -> int: ...
    @property
    def ndim(self) -> int: ...
    @property
    def numiter(self) -> int: ...
    @property
    def shape(self) -> _Shape: ...
    @property
    def size(self) -> int: ...
    def __next__(self) -> Tuple[Any, ...]: ...
    def __iter__(self: _T) -> _T: ...
    def reset(self) -> None: ...
