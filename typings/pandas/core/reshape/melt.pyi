from typing import TYPE_CHECKING
from pandas import DataFrame
from pandas.core.shared_docs import _shared_docs
from pandas.util._decorators import Appender, deprecate_kwarg

if TYPE_CHECKING: ...

@Appender(_shared_docs["melt"] % {"caller": "pd.melt(df, ", "other": "DataFrame.melt"})
def melt(
    frame: DataFrame,
    id_vars=...,
    value_vars=...,
    var_name=...,
    value_name=...,
    col_level=...,
    ignore_index: bool = ...,
) -> DataFrame: ...
@deprecate_kwarg(old_arg_name="label", new_arg_name=None)
def lreshape(data: DataFrame, groups, dropna: bool = ..., label=...) -> DataFrame: ...
def wide_to_long(
    df: DataFrame, stubnames, i, j, sep: str = ..., suffix: str = ...
) -> DataFrame: ...
