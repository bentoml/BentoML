from typing import TYPE_CHECKING
from pandas import DataFrame
from pandas._typing import AggFuncType, IndexLabel
from pandas.core.frame import _shared_docs
from pandas.util._decorators import Appender, Substitution

if TYPE_CHECKING: ...

@Substitution("\ndata : DataFrame")
@Appender(_shared_docs["pivot_table"], indents=1)
def pivot_table(
    data: DataFrame,
    values=...,
    index=...,
    columns=...,
    aggfunc: AggFuncType = ...,
    fill_value=...,
    margins=...,
    dropna=...,
    margins_name=...,
    observed=...,
    sort=...,
) -> DataFrame: ...
@Substitution("\ndata : DataFrame")
@Appender(_shared_docs["pivot"], indents=1)
def pivot(
    data: DataFrame,
    index: IndexLabel | None = ...,
    columns: IndexLabel | None = ...,
    values: IndexLabel | None = ...,
) -> DataFrame: ...
def crosstab(
    index,
    columns,
    values=...,
    rownames=...,
    colnames=...,
    aggfunc=...,
    margins=...,
    margins_name: str = ...,
    dropna: bool = ...,
    normalize=...,
) -> DataFrame: ...
