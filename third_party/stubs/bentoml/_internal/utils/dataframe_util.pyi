from bentoml.exceptions import BadInput as BadInput
from bentoml.utils import catch_exceptions as catch_exceptions
from bentoml.utils.csv import csv_quote as csv_quote, csv_row as csv_row, csv_split as csv_split, csv_splitlines as csv_splitlines, csv_unquote as csv_unquote
from bentoml.utils.lazy_loader import LazyLoader as LazyLoader
from typing import Any, Iterable, Mapping

pandas: Any

def check_dataframe_column_contains(required_column_names, df) -> None: ...
def guess_orient(table, strict: bool = ...): ...

class DataFrameState:
    columns: Any
    def __init__(self, columns: Mapping[str, int] = ...) -> None: ...

PANDAS_DATAFRAME_TO_JSON_ORIENT_OPTIONS: Any

def read_dataframes_from_json_n_csv(datas: Iterable[str], formats: Iterable[str], orient: str = ..., columns: Any | None = ..., dtype: Any | None = ...) -> Tuple[pandas.DataFrame, Iterable[slice]]: ...
