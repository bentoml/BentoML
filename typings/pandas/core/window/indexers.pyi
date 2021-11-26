import numpy as np
from pandas.util._decorators import Appender

get_window_bounds_doc = ...

class BaseIndexer:
    def __init__(
        self, index_array: np.ndarray | None = ..., window_size: int = ..., **kwargs
    ) -> None: ...
    @Appender(get_window_bounds_doc)
    def get_window_bounds(
        self,
        num_values: int = ...,
        min_periods: int | None = ...,
        center: bool | None = ...,
        closed: str | None = ...,
    ) -> tuple[np.ndarray, np.ndarray]: ...

class FixedWindowIndexer(BaseIndexer):
    @Appender(get_window_bounds_doc)
    def get_window_bounds(
        self,
        num_values: int = ...,
        min_periods: int | None = ...,
        center: bool | None = ...,
        closed: str | None = ...,
    ) -> tuple[np.ndarray, np.ndarray]: ...

class VariableWindowIndexer(BaseIndexer):
    @Appender(get_window_bounds_doc)
    def get_window_bounds(
        self,
        num_values: int = ...,
        min_periods: int | None = ...,
        center: bool | None = ...,
        closed: str | None = ...,
    ) -> tuple[np.ndarray, np.ndarray]: ...

class VariableOffsetWindowIndexer(BaseIndexer):
    def __init__(
        self,
        index_array: np.ndarray | None = ...,
        window_size: int = ...,
        index=...,
        offset=...,
        **kwargs
    ) -> None: ...
    @Appender(get_window_bounds_doc)
    def get_window_bounds(
        self,
        num_values: int = ...,
        min_periods: int | None = ...,
        center: bool | None = ...,
        closed: str | None = ...,
    ) -> tuple[np.ndarray, np.ndarray]: ...

class ExpandingIndexer(BaseIndexer):
    @Appender(get_window_bounds_doc)
    def get_window_bounds(
        self,
        num_values: int = ...,
        min_periods: int | None = ...,
        center: bool | None = ...,
        closed: str | None = ...,
    ) -> tuple[np.ndarray, np.ndarray]: ...

class FixedForwardWindowIndexer(BaseIndexer):
    @Appender(get_window_bounds_doc)
    def get_window_bounds(
        self,
        num_values: int = ...,
        min_periods: int | None = ...,
        center: bool | None = ...,
        closed: str | None = ...,
    ) -> tuple[np.ndarray, np.ndarray]: ...

class GroupbyIndexer(BaseIndexer):
    def __init__(
        self,
        index_array: np.ndarray | None = ...,
        window_size: int | BaseIndexer = ...,
        groupby_indices: dict | None = ...,
        window_indexer: type[BaseIndexer] = ...,
        indexer_kwargs: dict | None = ...,
        **kwargs
    ) -> None: ...
    @Appender(get_window_bounds_doc)
    def get_window_bounds(
        self,
        num_values: int = ...,
        min_periods: int | None = ...,
        center: bool | None = ...,
        closed: str | None = ...,
    ) -> tuple[np.ndarray, np.ndarray]: ...

class ExponentialMovingWindowIndexer(BaseIndexer):
    @Appender(get_window_bounds_doc)
    def get_window_bounds(
        self,
        num_values: int = ...,
        min_periods: int | None = ...,
        center: bool | None = ...,
        closed: str | None = ...,
    ) -> tuple[np.ndarray, np.ndarray]: ...
