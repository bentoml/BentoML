

from enum import Enum
from typing import Any, Callable, Generator, Literal, overload

import numpy as np
from pandas._typing import ArrayLike, DtypeObj

ndarray_obj_2d = np.ndarray
class NoDefault(Enum):
    ...


no_default: NoDefault
i8max: int
u8max: int
def item_from_zerodim(val: object) -> object:
    ...

def infer_dtype(value: object, skipna: bool = ...) -> str:
    ...

def is_iterator(obj: object) -> bool:
    ...

def is_scalar(val: object) -> bool:
    ...

def is_list_like(obj: object, allow_sets: bool = ...) -> bool:
    ...

def is_period(val: object) -> bool:
    ...

def is_interval(val: object) -> bool:
    ...

def is_decimal(val: object) -> bool:
    ...

def is_complex(val: object) -> bool:
    ...

def is_bool(val: object) -> bool:
    ...

def is_integer(val: object) -> bool:
    ...

def is_float(val: object) -> bool:
    ...

def is_interval_array(values: np.ndarray) -> bool:
    ...

def is_datetime64_array(values: np.ndarray) -> bool:
    ...

def is_timedelta_or_timedelta64_array(values: np.ndarray) -> bool:
    ...

def is_datetime_with_singletz_array(values: np.ndarray) -> bool:
    ...

def is_time_array(values: np.ndarray, skipna: bool = ...):
    ...

def is_date_array(values: np.ndarray, skipna: bool = ...):
    ...

def is_datetime_array(values: np.ndarray, skipna: bool = ...):
    ...

def is_string_array(values: np.ndarray, skipna: bool = ...):
    ...

def is_float_array(values: np.ndarray, skipna: bool = ...):
    ...

def is_integer_array(values: np.ndarray, skipna: bool = ...):
    ...

def is_bool_array(values: np.ndarray, skipna: bool = ...):
    ...

def fast_multiget(mapping: dict, keys: np.ndarray, default=...) -> np.ndarray:
    ...

def fast_unique_multiple_list_gen(gen: Generator, sort: bool = ...) -> list:
    ...

def fast_unique_multiple_list(lists: list, sort: bool = ...) -> list:
    ...

def fast_unique_multiple(arrays: list, sort: bool = ...) -> list:
    ...

def map_infer(arr: np.ndarray, f: Callable[[Any], Any], convert: bool = ..., ignore_na: bool = ...) -> np.ndarray:
    ...

@overload
def maybe_convert_objects(objects: np.ndarray, *, try_float: bool = ..., safe: bool = ..., convert_datetime: Literal[False] = ..., convert_timedelta: bool = ..., convert_period: Literal[False] = ..., convert_interval: Literal[False] = ..., convert_to_nullable_integer: Literal[False] = ..., dtype_if_all_nat: DtypeObj | None = ...) -> np.ndarray:
    ...

@overload
def maybe_convert_objects(objects: np.ndarray, *, try_float: bool = ..., safe: bool = ..., convert_datetime: bool = ..., convert_timedelta: bool = ..., convert_period: bool = ..., convert_interval: bool = ..., convert_to_nullable_integer: Literal[True] = ..., dtype_if_all_nat: DtypeObj | None = ...) -> ArrayLike:
    ...

@overload
def maybe_convert_objects(objects: np.ndarray, *, try_float: bool = ..., safe: bool = ..., convert_datetime: Literal[True] = ..., convert_timedelta: bool = ..., convert_period: bool = ..., convert_interval: bool = ..., convert_to_nullable_integer: bool = ..., dtype_if_all_nat: DtypeObj | None = ...) -> ArrayLike:
    ...

@overload
def maybe_convert_objects(objects: np.ndarray, *, try_float: bool = ..., safe: bool = ..., convert_datetime: bool = ..., convert_timedelta: bool = ..., convert_period: Literal[True] = ..., convert_interval: bool = ..., convert_to_nullable_integer: bool = ..., dtype_if_all_nat: DtypeObj | None = ...) -> ArrayLike:
    ...

@overload
def maybe_convert_objects(objects: np.ndarray, *, try_float: bool = ..., safe: bool = ..., convert_datetime: bool = ..., convert_timedelta: bool = ..., convert_period: bool = ..., convert_interval: bool = ..., convert_to_nullable_integer: bool = ..., dtype_if_all_nat: DtypeObj | None = ...) -> ArrayLike:
    ...

@overload
def maybe_convert_numeric(values: np.ndarray, na_values: set, convert_empty: bool = ..., coerce_numeric: bool = ..., convert_to_masked_nullable: Literal[False] = ...) -> tuple[np.ndarray, None]:
    ...

@overload
def maybe_convert_numeric(values: np.ndarray, na_values: set, convert_empty: bool = ..., coerce_numeric: bool = ..., *, convert_to_masked_nullable: Literal[True]) -> tuple[np.ndarray, np.ndarray]:
    ...

def ensure_string_array(arr, na_value: object = ..., convert_na_value: bool = ..., copy: bool = ..., skipna: bool = ...) -> np.ndarray:
    ...

def infer_datetimelike_array(arr: np.ndarray) -> tuple[str, bool]:
    ...

def astype_intsafe(arr: np.ndarray, new_dtype: np.dtype) -> np.ndarray:
    ...

def fast_zip(ndarrays: list) -> np.ndarray:
    ...

def to_object_array_tuples(rows: object) -> ndarray_obj_2d:
    ...

def tuples_to_object_array(tuples: np.ndarray) -> ndarray_obj_2d:
    ...

def to_object_array(rows: object, min_width: int = ...) -> ndarray_obj_2d:
    ...

def dicts_to_array(dicts: list, columns: list) -> ndarray_obj_2d:
    ...

def maybe_booleans_to_slice(mask: np.ndarray) -> slice | np.ndarray:
    ...

def maybe_indices_to_slice(indices: np.ndarray, max_len: int) -> slice | np.ndarray:
    ...

def is_all_arraylike(obj: list) -> bool:
    ...

def memory_usage_of_objects(arr: np.ndarray) -> int:
    ...

def map_infer_mask(arr: np.ndarray, f: Callable[[Any], Any], mask: np.ndarray, convert: bool = ..., na_value: Any = ..., dtype: np.dtype = ...) -> np.ndarray:
    ...

def indices_fast(index: np.ndarray, labels: np.ndarray, keys: list, sorted_labels: list[np.ndarray]) -> dict:
    ...

def generate_slices(labels: np.ndarray, ngroups: int) -> tuple[np.ndarray, np.ndarray],:
    ...

def count_level_2d(mask: np.ndarray, labels: np.ndarray, max_bin: int, axis: int) -> np.ndarray:
    ...

def get_level_sorter(label: np.ndarray, starts: np.ndarray) -> np.ndarray:
    ...

def generate_bins_dt64(values: np.ndarray, binner: np.ndarray, closed: object = ..., hasnans: bool = ...) -> np.ndarray:
    ...

def array_equivalent_object(left: np.ndarray, right: np.ndarray) -> bool:
    ...

def has_infs_f8(arr: np.ndarray) -> bool:
    ...

def has_infs_f4(arr: np.ndarray) -> bool:
    ...

def get_reverse_indexer(indexer: np.ndarray, length: int) -> np.ndarray:
    ...

def is_bool_list(obj: list) -> bool:
    ...

