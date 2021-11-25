import abc
from typing import TYPE_CHECKING, Any, Dict, Iterator

from pandas import DataFrame, Index, Series
from pandas._typing import (
    AggFuncType,
    AggFuncTypeDict,
    AggObjType,
    Axis,
    FrameOrSeries,
    FrameOrSeriesUnion,
)
from pandas.core.groupby import GroupBy
from pandas.core.resample import Resampler
from pandas.core.window.rolling import BaseWindow
from pandas.util._decorators import cache_readonly

if TYPE_CHECKING: ...
ResType = Dict[int, Any]

def frame_apply(
    obj: DataFrame,
    func: AggFuncType,
    axis: Axis = ...,
    raw: bool = ...,
    result_type: str | None = ...,
    args=...,
    kwargs=...,
) -> FrameApply:
    """construct and return a row or column based frame apply object"""
    ...

class Apply(metaclass=abc.ABCMeta):
    axis: int
    def __init__(
        self, obj: AggObjType, func, raw: bool, result_type: str | None, args, kwargs
    ) -> None: ...
    @abc.abstractmethod
    def apply(self) -> FrameOrSeriesUnion: ...
    def agg(self) -> FrameOrSeriesUnion | None:
        """
        Provide an implementation for the aggregators.

        Returns
        -------
        Result of aggregation, or None if agg cannot be performed by
        this method.
        """
        ...
    def transform(self) -> FrameOrSeriesUnion:
        """
        Transform a DataFrame or Series.

        Returns
        -------
        DataFrame or Series
            Result of applying ``func`` along the given axis of the
            Series or DataFrame.

        Raises
        ------
        ValueError
            If the transform function fails or does not transform.
        """
        ...
    def transform_dict_like(self, func):
        """
        Compute transform in the case of a dict-like func
        """
        ...
    def transform_str_or_callable(self, func) -> FrameOrSeriesUnion:
        """
        Compute transform in the case of a string or callable func
        """
        ...
    def agg_list_like(self) -> FrameOrSeriesUnion:
        """
        Compute aggregation in the case of a list-like argument.

        Returns
        -------
        Result of aggregation.
        """
        ...
    def agg_dict_like(self) -> FrameOrSeriesUnion:
        """
        Compute aggregation in the case of a dict-like argument.

        Returns
        -------
        Result of aggregation.
        """
        ...
    def apply_str(self) -> FrameOrSeriesUnion:
        """
        Compute apply in case of a string.

        Returns
        -------
        result: Series or DataFrame
        """
        ...
    def apply_multiple(self) -> FrameOrSeriesUnion:
        """
        Compute apply in case of a list-like or dict-like.

        Returns
        -------
        result: Series, DataFrame, or None
            Result when self.f is a list-like or dict-like, None otherwise.
        """
        ...
    def normalize_dictlike_arg(
        self, how: str, obj: FrameOrSeriesUnion, func: AggFuncTypeDict
    ) -> AggFuncTypeDict:
        """
        Handler for dict-like argument.

        Ensures that necessary columns exist if obj is a DataFrame, and
        that a nested renamer is not passed. Also normalizes to all lists
        when values consists of a mix of list and non-lists.
        """
        ...

class NDFrameApply(Apply):
    """
    Methods shared by FrameApply and SeriesApply but
    not GroupByApply or ResamplerWindowApply
    """

    @property
    def index(self) -> Index: ...
    @property
    def agg_axis(self) -> Index: ...

class FrameApply(NDFrameApply):
    obj: DataFrame
    @property
    @abc.abstractmethod
    def result_index(self) -> Index: ...
    @property
    @abc.abstractmethod
    def result_columns(self) -> Index: ...
    @property
    @abc.abstractmethod
    def series_generator(self) -> Iterator[Series]: ...
    @abc.abstractmethod
    def wrap_results_for_axis(
        self, results: ResType, res_index: Index
    ) -> FrameOrSeriesUnion: ...
    @property
    def res_columns(self) -> Index: ...
    @property
    def columns(self) -> Index: ...
    @cache_readonly
    def values(self): ...
    @cache_readonly
    def dtypes(self) -> Series: ...
    def apply(self) -> FrameOrSeriesUnion:
        """compute the results"""
        ...
    def agg(self): ...
    def apply_empty_result(self):  # -> DataFrame | Series:
        """
        we have an empty result; at least 1 axis is 0

        we will try to apply the function to an empty
        series in order to see if this is a reduction function
        """
        ...
    def apply_raw(self):  # -> DataFrame | Series:
        """apply to the values as a numpy array"""
        ...
    def apply_broadcast(self, target: DataFrame) -> DataFrame: ...
    def apply_standard(self): ...
    def apply_series_generator(self) -> tuple[ResType, Index]: ...
    def wrap_results(self, results: ResType, res_index: Index) -> FrameOrSeriesUnion: ...
    def apply_str(self) -> FrameOrSeriesUnion: ...

class FrameRowApply(FrameApply):
    axis = ...
    def apply_broadcast(self, target: DataFrame) -> DataFrame: ...
    @property
    def series_generator(self): ...
    @property
    def result_index(self) -> Index: ...
    @property
    def result_columns(self) -> Index: ...
    def wrap_results_for_axis(
        self, results: ResType, res_index: Index
    ) -> FrameOrSeriesUnion:
        """return the results for the rows"""
        ...

class FrameColumnApply(FrameApply):
    axis = ...
    def apply_broadcast(self, target: DataFrame) -> DataFrame: ...
    @property
    def series_generator(self): ...
    @property
    def result_index(self) -> Index: ...
    @property
    def result_columns(self) -> Index: ...
    def wrap_results_for_axis(
        self, results: ResType, res_index: Index
    ) -> FrameOrSeriesUnion:
        """return the results for the columns"""
        ...
    def infer_to_same_shape(self, results: ResType, res_index: Index) -> DataFrame:
        """infer the results to the same shape as the input object"""
        ...

class SeriesApply(NDFrameApply):
    obj: Series
    axis = ...
    def __init__(
        self, obj: Series, func: AggFuncType, convert_dtype: bool, args, kwargs
    ) -> None: ...
    def apply(self) -> FrameOrSeriesUnion: ...
    def agg(self): ...
    def apply_empty_result(self) -> Series: ...
    def apply_standard(self) -> FrameOrSeriesUnion: ...

class GroupByApply(Apply):
    def __init__(
        self, obj: GroupBy[FrameOrSeries], func: AggFuncType, args, kwargs
    ) -> None: ...
    def apply(self): ...
    def transform(self): ...

class ResamplerWindowApply(Apply):
    axis = ...
    obj: Resampler | BaseWindow
    def __init__(
        self, obj: Resampler | BaseWindow, func: AggFuncType, args, kwargs
    ) -> None: ...
    def apply(self): ...
    def transform(self): ...
