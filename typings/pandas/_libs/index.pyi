

import numpy as np

class IndexEngine:
    over_size_threshold: bool
    def __init__(self, vgetter, n: int) -> None:
        ...
    
    def __contains__(self, val: object) -> bool:
        ...
    
    def get_loc(self, val: object) -> int | slice | np.ndarray:
        ...
    
    def sizeof(self, deep: bool = ...) -> int:
        ...
    
    def __sizeof__(self) -> int:
        ...
    
    @property
    def is_unique(self) -> bool:
        ...
    
    @property
    def is_monotonic_increasing(self) -> bool:
        ...
    
    @property
    def is_monotonic_decreasing(self) -> bool:
        ...
    
    def get_backfill_indexer(self, other: np.ndarray, limit: int | None = ...) -> np.ndarray:
        ...
    
    def get_pad_indexer(self, other: np.ndarray, limit: int | None = ...) -> np.ndarray:
        ...
    
    @property
    def is_mapping_populated(self) -> bool:
        ...
    
    def clear_mapping(self):
        ...
    
    def get_indexer(self, values: np.ndarray) -> np.ndarray:
        ...
    
    def get_indexer_non_unique(self, targets: np.ndarray) -> tuple[np.ndarray, np.ndarray],:
        ...
    


class Float64Engine(IndexEngine):
    ...


class Float32Engine(IndexEngine):
    ...


class Int64Engine(IndexEngine):
    ...


class Int32Engine(IndexEngine):
    ...


class Int16Engine(IndexEngine):
    ...


class Int8Engine(IndexEngine):
    ...


class UInt64Engine(IndexEngine):
    ...


class UInt32Engine(IndexEngine):
    ...


class UInt16Engine(IndexEngine):
    ...


class UInt8Engine(IndexEngine):
    ...


class ObjectEngine(IndexEngine):
    ...


class DatetimeEngine(Int64Engine):
    ...


class TimedeltaEngine(DatetimeEngine):
    ...


class PeriodEngine(Int64Engine):
    ...


class BaseMultiIndexCodesEngine:
    levels: list[np.ndarray]
    offsets: np.ndarray
    def __init__(self, levels: list[np.ndarray], labels: list[np.ndarray], offsets: np.ndarray) -> None:
        ...
    
    def get_indexer(self, target: np.ndarray) -> np.ndarray:
        ...
    
    def get_indexer_with_fill(self, target: np.ndarray, values: np.ndarray, method: str, limit: int | None) -> np.ndarray:
        ...
    


