from datetime import datetime, timedelta, tzinfo
from io import BufferedIOBase, RawIOBase, TextIOBase, TextIOWrapper
from mmap import mmap
from os import PathLike
from typing import (
    IO,
    Any,
    AnyStr,
    Callable,
    Collection,
    Dict,
    Hashable,
    List,
    Literal,
    Mapping,
    Optional,
    Sequence,
    Tuple,
)
from typing import Type as type_t
from typing import TypeVar, Union
import numpy as np
import numpy.typing as npt
from pandas._libs import Period, Timedelta, Timestamp
from pandas.core.arrays.base import ExtensionArray
from pandas.core.dtypes.dtypes import ExtensionDtype
from pandas.core.frame import DataFrame
from pandas.core.groupby.generic import DataFrameGroupBy, GroupBy, SeriesGroupBy
from pandas.core.indexes.base import Index
from pandas.core.resample import Resampler
from pandas.core.series import Series
from pandas.core.window.rolling import BaseWindow
from pandas.io.formats.format import EngFormatter

Interval = Any
DateOffset = Any
ArrayManager = Any
BlockManager = Any
SingleArrayManager = Any
SingleBlockManager = Any
ArrayLike = Union["ExtensionArray", np.ndarray[Any, np.dtype[Any]]]
AnyArrayLike = Union[ArrayLike, "Index", "Series"]
PythonScalar = Union[str, int, float, bool]
DatetimeLikeScalar = Union["Period", "Timestamp", "Timedelta"]
PandasScalar = Union["Period", "Timestamp", "Timedelta", "Interval"]
Scalar = Union[PythonScalar, PandasScalar]
TimestampConvertibleTypes = Union[
    "Timestamp", datetime, np.datetime64, int, np.int64, float, str
]
TimedeltaConvertibleTypes = Union[
    "Timedelta", timedelta, np.timedelta64, int, np.int64, float, str
]
Timezone = Union[str, tzinfo]
FrameOrSeriesUnion = Union["DataFrame", "Series"]
Axis = Union[str, int]
IndexLabel = Union[Hashable, Sequence[Hashable]]
Level = Union[Hashable, int]
Shape = Tuple[int, ...]
Suffixes = Tuple[str, str]
Ordered = Optional[bool]
JSONSerializable = Optional[
    Union[PythonScalar, List[str], Dict[str, Union[str, bytes]]]
]
Frequency = Union[str, "DateOffset"]
Axes = Collection[Any]
NpDtype = Union[str, np.dtype[Any]]
Dtype = Union[
    "ExtensionDtype", NpDtype, type_t[Union[str, float, int, complex, bool, object]]
]
DtypeArg = Union[Dtype, Dict[Hashable, Dtype]]
DtypeObj = Union[np.dtype[Any], "ExtensionDtype"]
Renamer = Union[Mapping[Hashable, Any], Callable[[Hashable], Hashable]]
T = TypeVar("T")
FuncType = Callable[..., Any]
F = TypeVar("F")
ValueKeyFunc = Optional[Callable[["Series"], Union["Series", AnyArrayLike]]]
IndexKeyFunc = Optional[Callable[["Index"], Union["Index", AnyArrayLike]]]
AggFuncTypeBase = Union[Callable[..., Any], str]
AggFuncTypeDict = Dict[Hashable, Union[AggFuncTypeBase, List[AggFuncTypeBase]]]
AggFuncType = Union[AggFuncTypeBase, List[AggFuncTypeBase], AggFuncTypeDict]
AggObjType = Union[
    "Series",
    "DataFrame",
    "GroupBy",
    "SeriesGroupBy",
    "DataFrameGroupBy",
    "BaseWindow",
    "Resampler",
]
PythonFuncType = Callable[[Any], Any]
Buffer = Union[IO[AnyStr], RawIOBase, BufferedIOBase, TextIOBase, TextIOWrapper, mmap]
FileOrBuffer = Union[str, Buffer[AnyStr]]
FilePathOrBuffer = Union["PathLike[str]", FileOrBuffer[AnyStr]]
StorageOptions = Optional[Dict[str, Any]]
CompressionDict = Dict[str, Any]
CompressionOptions = Optional[Union[str, CompressionDict]]
FormattersType = Union[
    List[Callable[..., Any]],
    Tuple[Callable[..., Any], ...],
    Mapping[Union[str, int], Callable[..., Any]],
]
ColspaceType = Mapping[Hashable, Union[str, int]]
FloatFormatType = Union[str, Callable[..., Any], "EngFormatter"]
ColspaceArgType = Union[
    str, int, Sequence[Union[str, int]], Mapping[Hashable, Union[str, int]]
]
FillnaOptions = Literal["backfill", "bfill", "ffill", "pad"]
Manager = Union[
    "ArrayManager", "SingleArrayManager", "BlockManager", "SingleBlockManager"
]
SingleManager = Union["SingleArrayManager", "SingleBlockManager"]
Manager2D = Union["ArrayManager", "BlockManager"]
PositionalIndexer = Union[
    int, np.integer[Any], slice, Sequence[int], np.ndarray[Any, np.dtype[Any]]
]
PositionalIndexer2D = Union[
    PositionalIndexer, Tuple[PositionalIndexer, PositionalIndexer]
]
WindowingRankType = Literal["average", "min", "max"]
