from datetime import datetime, timedelta, tzinfo
from io import BufferedIOBase, RawIOBase, TextIOBase, TextIOWrapper
from mmap import mmap
from typing import (
    IO,
    TYPE_CHECKING,
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
from typing import Union

import numpy as np

if TYPE_CHECKING: ...
else: ...
ArrayLike = Union["ExtensionArray", np.ndarray]
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
FrameOrSeries = ...
Axis = Union[str, int]
IndexLabel = Union[Hashable, Sequence[Hashable]]
Level = Union[Hashable, int]
Shape = Tuple[int, ...]
Suffixes = Tuple[str, str]
Ordered = Optional[bool]
JSONSerializable = Optional[Union[PythonScalar, List, Dict]]
Frequency = Union[str, "DateOffset"]
Axes = Collection[Any]
NpDtype = Union[str, np.dtype]
Dtype = Union[
    "ExtensionDtype", NpDtype, type_t[Union[str, float, int, complex, bool, object]]
]
DtypeArg = Union[Dtype, Dict[Hashable, Dtype]]
DtypeObj = Union[np.dtype, "ExtensionDtype"]
Renamer = Union[Mapping[Hashable, Any], Callable[[Hashable], Hashable]]
T = ...
FuncType = Callable[..., Any]
F = ...
ValueKeyFunc = Optional[Callable[["Series"], Union["Series", AnyArrayLike]]]
IndexKeyFunc = Optional[Callable[["Index"], Union["Index", AnyArrayLike]]]
AggFuncTypeBase = Union[Callable, str]
AggFuncTypeDict = Dict[Hashable, Union[AggFuncTypeBase, List[AggFuncTypeBase]]]
AggFuncType = (Union[AggFuncTypeBase, List[AggFuncTypeBase], AggFuncTypeDict],)
AggObjType = (
    Union[
        "Series",
        "DataFrame",
        "GroupBy",
        "SeriesGroupBy",
        "DataFrameGroupBy",
        "BaseWindow",
        "Resampler",
    ],
)
PythonFuncType = Callable[[Any], Any]
Buffer = Union[IO[AnyStr], RawIOBase, BufferedIOBase, TextIOBase, TextIOWrapper, mmap]
FileOrBuffer = Union[str, Buffer[AnyStr]]
FilePathOrBuffer = Union["PathLike[str]", FileOrBuffer[AnyStr]]
StorageOptions = Optional[Dict[str, Any]]
CompressionDict = Dict[str, Any]
CompressionOptions = Optional[Union[str, CompressionDict]]
FormattersType = Union[
    List[Callable], Tuple[Callable, ...], Mapping[Union[str, int], Callable]
]
ColspaceType = Mapping[Hashable, Union[str, int]]
FloatFormatType = Union[str, Callable, "EngFormatter"]
ColspaceArgType = Union[
    str, int, Sequence[Union[str, int]], Mapping[Hashable, Union[str, int]]
]
if TYPE_CHECKING:
    FillnaOptions = Literal["backfill", "bfill", "ffill", "pad"]
else: ...
Manager = Union[
    "ArrayManager", "SingleArrayManager", "BlockManager", "SingleBlockManager"
]
SingleManager = Union["SingleArrayManager", "SingleBlockManager"]
Manager2D = Union["ArrayManager", "BlockManager"]
PositionalIndexer = Union[int, np.integer, slice, Sequence[int], np.ndarray]
PositionalIndexer2D = Union[
    PositionalIndexer, Tuple[PositionalIndexer, PositionalIndexer]
]
