from typing import TYPE_CHECKING, List, Tuple, Union, overload
import numpy as np
from pandas import Series
from pandas._libs.tslibs.nattype import NaTType
from pandas._typing import AnyArrayLike
from pandas.core.indexes.datetimes import DatetimeIndex

if TYPE_CHECKING: ...
ArrayConvertible = Union[List, Tuple, AnyArrayLike, "Series"]
Scalar = Union[int, float, str]
DatetimeScalar = ...
DatetimeScalarOrArrayConvertible = Union[DatetimeScalar, ArrayConvertible]
start_caching_at = ...

def should_cache(
    arg: ArrayConvertible, unique_share: float = ..., check_count: int | None = ...
) -> bool: ...
@overload
def to_datetime(
    arg: DatetimeScalar,
    errors: str = ...,
    dayfirst: bool = ...,
    yearfirst: bool = ...,
    utc: bool | None = ...,
    format: str | None = ...,
    exact: bool = ...,
    unit: str | None = ...,
    infer_datetime_format: bool = ...,
    origin=...,
    cache: bool = ...,
) -> DatetimeScalar | NaTType: ...
@overload
def to_datetime(
    arg: Series,
    errors: str = ...,
    dayfirst: bool = ...,
    yearfirst: bool = ...,
    utc: bool | None = ...,
    format: str | None = ...,
    exact: bool = ...,
    unit: str | None = ...,
    infer_datetime_format: bool = ...,
    origin=...,
    cache: bool = ...,
) -> Series: ...
@overload
def to_datetime(
    arg: list | tuple | np.ndarray,
    errors: str = ...,
    dayfirst: bool = ...,
    yearfirst: bool = ...,
    utc: bool | None = ...,
    format: str | None = ...,
    exact: bool = ...,
    unit: str | None = ...,
    infer_datetime_format: bool = ...,
    origin=...,
    cache: bool = ...,
) -> DatetimeIndex: ...
def to_datetime(
    arg: DatetimeScalarOrArrayConvertible,
    errors: str = ...,
    dayfirst: bool = ...,
    yearfirst: bool = ...,
    utc: bool | None = ...,
    format: str | None = ...,
    exact: bool = ...,
    unit: str | None = ...,
    infer_datetime_format: bool = ...,
    origin=...,
    cache: bool = ...,
) -> DatetimeIndex | Series | DatetimeScalar | NaTType | None: ...

_unit_map = ...
