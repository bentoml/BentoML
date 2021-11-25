from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Callable, Sequence

from pandas import DataFrame, Series
from pandas._typing import FrameOrSeries, FrameOrSeriesUnion, Hashable

"""
Module responsible for execution of NDFrame.describe() method.

Method NDFrame.describe() delegates actual execution to function describe_ndframe().
"""
if TYPE_CHECKING: ...

def describe_ndframe(
    *,
    obj: FrameOrSeries,
    include: str | Sequence[str] | None,
    exclude: str | Sequence[str] | None,
    datetime_is_numeric: bool,
    percentiles: Sequence[float] | None
) -> FrameOrSeries:
    """Describe series or dataframe.

    Called from pandas.core.generic.NDFrame.describe()

    Parameters
    ----------
    obj: DataFrame or Series
        Either dataframe or series to be described.
    include : 'all', list-like of dtypes or None (default), optional
        A white list of data types to include in the result. Ignored for ``Series``.
    exclude : list-like of dtypes or None (default), optional,
        A black list of data types to omit from the result. Ignored for ``Series``.
    datetime_is_numeric : bool, default False
        Whether to treat datetime dtypes as numeric.
    percentiles : list-like of numbers, optional
        The percentiles to include in the output. All should fall between 0 and 1.
        The default is ``[.25, .5, .75]``, which returns the 25th, 50th, and
        75th percentiles.

    Returns
    -------
    Dataframe or series description.
    """
    ...

class NDFrameDescriberAbstract(ABC):
    """Abstract class for describing dataframe or series.

    Parameters
    ----------
    obj : Series or DataFrame
        Object to be described.
    datetime_is_numeric : bool
        Whether to treat datetime dtypes as numeric.
    """

    def __init__(self, obj: FrameOrSeriesUnion, datetime_is_numeric: bool) -> None: ...
    @abstractmethod
    def describe(self, percentiles: Sequence[float]) -> FrameOrSeriesUnion:
        """Do describe either series or dataframe.

        Parameters
        ----------
        percentiles : list-like of numbers
            The percentiles to include in the output.
        """
        ...

class SeriesDescriber(NDFrameDescriberAbstract):
    """Class responsible for creating series description."""

    obj: Series
    def describe(self, percentiles: Sequence[float]) -> Series: ...

class DataFrameDescriber(NDFrameDescriberAbstract):
    """Class responsible for creating dataobj description.

    Parameters
    ----------
    obj : DataFrame
        DataFrame to be described.
    include : 'all', list-like of dtypes or None
        A white list of data types to include in the result.
    exclude : list-like of dtypes or None
        A black list of data types to omit from the result.
    datetime_is_numeric : bool
        Whether to treat datetime dtypes as numeric.
    """

    def __init__(
        self,
        obj: DataFrame,
        *,
        include: str | Sequence[str] | None,
        exclude: str | Sequence[str] | None,
        datetime_is_numeric: bool
    ) -> None: ...
    def describe(self, percentiles: Sequence[float]) -> DataFrame: ...

def reorder_columns(ldesc: Sequence[Series]) -> list[Hashable]:
    """Set a convenient order for rows for display."""
    ...

def describe_numeric_1d(series: Series, percentiles: Sequence[float]) -> Series:
    """Describe series containing numerical data.

    Parameters
    ----------
    series : Series
        Series to be described.
    percentiles : list-like of numbers
        The percentiles to include in the output.
    """
    ...

def describe_categorical_1d(data: Series, percentiles_ignored: Sequence[float]) -> Series:
    """Describe series containing categorical data.

    Parameters
    ----------
    data : Series
        Series to be described.
    percentiles_ignored : list-like of numbers
        Ignored, but in place to unify interface.
    """
    ...

def describe_timestamp_as_categorical_1d(
    data: Series, percentiles_ignored: Sequence[float]
) -> Series:
    """Describe series containing timestamp data treated as categorical.

    Parameters
    ----------
    data : Series
        Series to be described.
    percentiles_ignored : list-like of numbers
        Ignored, but in place to unify interface.
    """
    ...

def describe_timestamp_1d(data: Series, percentiles: Sequence[float]) -> Series:
    """Describe series containing datetime64 dtype.

    Parameters
    ----------
    data : Series
        Series to be described.
    percentiles : list-like of numbers
        The percentiles to include in the output.
    """
    ...

def select_describe_func(data: Series, datetime_is_numeric: bool) -> Callable:
    """Select proper function for describing series based on data type.

    Parameters
    ----------
    data : Series
        Series to be described.
    datetime_is_numeric : bool
        Whether to treat datetime dtypes as numeric.
    """
    ...

def refine_percentiles(percentiles: Sequence[float] | None) -> Sequence[float]:
    """Ensure that percentiles are unique and sorted.

    Parameters
    ----------
    percentiles : list-like of numbers, optional
        The percentiles to include in the output.
    """
    ...
