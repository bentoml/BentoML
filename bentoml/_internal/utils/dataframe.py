import io
import json
import typing as t
import itertools
from typing import TYPE_CHECKING

from . import catch_exceptions
from .csv import csv_row
from .csv import csv_quote
from .csv import csv_split
from .csv import csv_unquote
from .csv import csv_splitlines
from .lazy_loader import LazyLoader
from ...exceptions import BadInput
from ...exceptions import BentoMLException

if TYPE_CHECKING:
    import pandas as pd
else:
    pd = LazyLoader("pd", globals(), "pandas")


def check_dataframe_column_contains(
    required_column_names: str, df: "pd.DataFrame"
) -> None:
    df_columns = set(map(str, df.columns))
    for col in required_column_names:
        if col not in df_columns:
            raise BadInput(
                f"Missing columns: {','.join(set(required_column_names) - df_columns)}, required_column:{df_columns}"  # noqa: E501
            )


@catch_exceptions(Exception, BentoMLException, fallback=None)
def guess_orient(
    table: t.Union[t.List[t.Mapping[str, t.Any]], t.Dict[str, t.Any]],
    strict: bool = False,
) -> t.Optional[t.Set[str]]:
    if isinstance(table, list):
        if not table:
            if strict:
                return {"records", "values"}
            else:
                return {"records"}
        if isinstance(table[0], dict):
            return {"records"}
        else:
            return {"values"}
    elif isinstance(table, dict):
        if set(table) == {"columns", "index", "data"}:
            return {"split"}
        if set(table) == {"schema", "data"} and "primaryKey" in table["schema"]:
            return {"table"}
        if strict:
            return {"columns", "index"}
        else:
            return {"columns"}
    else:
        return None


class _DataFrameState(object):
    # fmt: off
    @t.overload
    def __init__(self, columns: t.Optional[t.Dict[str, int]]): ...  # noqa: F811,E704

    @t.overload  # noqa: F811
    def __init__(self, columns: t.Optional[t.Tuple[str, ...]]): ...  # noqa: F811,E704
    # fmt: on

    def __init__(  # noqa: F811
        self,
        columns: t.Optional[t.Union[t.Mapping[str, int], t.Tuple[str, ...]]] = None,
    ):
        self.columns = columns


def _from_json_records(state: _DataFrameState, table: list) -> t.Iterator[str]:
    if state.columns is None:  # make header
        state.columns = {k: i for i, k in enumerate(table[0].keys())}
    for tr in table:
        yield csv_row(tr[c] for c in state.columns)


def _from_json_values(_: _DataFrameState, table: list) -> t.Iterator[str]:
    for tr in table:
        yield csv_row(tr)


def _from_json_columns(state: _DataFrameState, table: dict) -> t.Iterator[str]:
    if state.columns is None:  # make header
        state.columns = {k: i for i, k in enumerate(table.keys())}
    for row in next(iter(table.values())):
        yield csv_row(table[col][row] for col in state.columns)


def _from_json_index(state: _DataFrameState, table: dict) -> t.Iterator[str]:
    if state.columns is None:  # make header
        state.columns = {k: i for i, k in enumerate(next(iter(table.values())).keys())}
        for row in table.keys():
            yield csv_row(td for td in table[row].values())
    else:
        for row in table.keys():
            yield csv_row(table[row][col] for col in state.columns)


def _from_json_split(state: _DataFrameState, table: dict) -> t.Iterator[str]:
    table_columns = {k: i for i, k in enumerate(table["columns"])}

    if state.columns is None:  # make header
        state.columns = table_columns
        for row in table["data"]:
            yield csv_row(row)
    else:
        idxs = [state.columns[k] for k in table_columns]
        for row in table["data"]:
            yield csv_row(row[idx] for idx in idxs)


def _from_csv_without_index(
    state: _DataFrameState, table: t.Iterator[str]
) -> t.Iterator[str]:
    row_str = next(table)  # skip column names
    table_columns = tuple(csv_unquote(s) for s in csv_split(row_str, ","))

    if state.columns is None:
        state.columns = table_columns
        for row_str in table:
            if not row_str:  # skip blank line
                continue
            if not row_str.strip():
                yield csv_quote(row_str)
            else:
                yield row_str
    elif not all(
        c1 == c2 for c1, c2 in itertools.zip_longest(state.columns, table_columns)
    ):
        # TODO: check type hint for this case. Right now nothing breaks so :)
        idxs = [state.columns[k] for k in table_columns]  # type: ignore[call-overload]
        for row_str in table:
            if not row_str:  # skip blank line
                continue
            if not row_str.strip():
                yield csv_quote(row_str)
            else:
                tr = tuple(s for s in csv_split(row_str, ","))
                yield csv_row(tr[i] for i in idxs)
    else:
        for row_str in table:
            if not row_str:  # skip blank line
                continue
            if not row_str.strip():
                yield csv_quote(row_str)
            else:
                yield row_str


_ORIENT_MAP: t.Dict[str, t.Callable[["_DataFrameState", str], t.Iterator[str]]] = {
    "records": _from_json_records,
    "columns": _from_json_columns,
    "values": _from_json_values,
    "split": _from_json_split,
    "index": _from_json_index,
    # 'table': _from_json_table,
}

PANDAS_DATAFRAME_TO_JSON_ORIENT_OPTIONS = {k for k in _ORIENT_MAP}


def _dataframe_csv_from_input(
    table: str,
    fmt: str,
    orient: t.Optional[str],
    state: _DataFrameState,
) -> t.Optional[t.Tuple[str, ...]]:
    try:
        if not fmt or fmt == "json":
            table = json.loads(table)
            if not orient:
                orient = guess_orient(table, strict=False).pop()
            else:
                # TODO: this can be either a set or a string
                guessed_orient = guess_orient(table, strict=True)  # type: t.Set[str]
                if set(orient) != guessed_orient and orient not in guessed_orient:
                    return None
            if orient not in _ORIENT_MAP:
                return None
            _from_json = _ORIENT_MAP[orient]
            try:
                return tuple(_from_json(state, table))
            except (TypeError, AttributeError, KeyError, IndexError):
                return None
        elif fmt == "csv":
            _table = csv_splitlines(table)
            return tuple(_from_csv_without_index(state, _table))
        else:
            return None
    except json.JSONDecodeError:
        return None


def from_json_or_csv(
    data: t.Iterable[str],
    formats: t.Iterable[str],
    orient: t.Optional[str] = None,
    columns: t.Optional[t.List[str]] = None,
    dtype: t.Optional[t.Union[bool, t.Dict[str, t.Any]]] = None,
) -> t.Tuple[t.Optional["pd.DataFrame"], t.Tuple[int, ...]]:
    """
    Load DataFrames from multiple raw data sources in JSON or CSV format, efficiently

    Background: Each call of `pandas.read_csv()` or `pandas.read_json` takes about
     100ms, no matter how many lines the read data contains. This function concats
     the ragged_tensor/csv before running `read_json`/`read_csv` to improve performance.

    Args:
        data (`Iterable[str]`):
            Data in either JSON or CSV format
        formats (`Iterable[str]`):
            List of formats, which are either `json` or `csv`
        orient (:code:`str`, `optional`, default `"records"`):
            Indication of expected JSON string format. Compatible JSON strings can be
             produced by `pandas.io.json.to_json()` with a corresponding orient value.
             Possible orients are:
                - `split` - :code:`Dict[str, Any]`: {idx -> [idx], columns -> [columns], data
                   -> [values]}
                - `records` - `List[Any]`: [{column -> value}, ..., {column -> value}]
                - `index` - :code:`Dict[str, Any]`: {idx -> {column -> value}}
                - `columns` - :code:`Dict[str, Any]`: {column -> {index -> value}}
                - `values` - :code:`Dict[str, Any]`: Values arrays
        columns (`List[str]`, `optional`, default `None`):
            List of column names that users wish to update
        dtype (:code:`Union[bool, Dict[str, Any]]`, `optional`, default `None`):
            Data type to inputs/outputs to. If it is a boolean, then pandas will infer
             data types. Otherwise, if it is a dictionary of column to data type, then
             applies those to incoming dataframes. If False, then don't infer data types
             at all (only applies to the data). This is not applicable when
             `orient='table'`.

    Returns:
        A tuple containing a `pandas.DataFrame` and a tuple containing the length of all
         series in the returned DataFrame.

    Raises:
        pandas.errors.EmptyDataError:
            When data is not found or is empty.
    """
    state = _DataFrameState(
        columns={k: i for i, k in enumerate(columns)} if columns else None
    )
    trs_list = tuple(
        _dataframe_csv_from_input(_t, _fmt, orient, state)
        for _t, _fmt in zip(data, formats)
    )
    header = ",".join(csv_quote(td) for td in state.columns) if state.columns else None
    lens = tuple(len(trs) if trs else 0 for trs in trs_list)
    table = "\n".join(tr for trs in trs_list if trs is not None for tr in trs)
    try:
        if not header:
            df = pd.read_csv(
                io.StringIO(table),
                dtype=dtype,
                index_col=None,
                header=None,
            )
        else:
            df = pd.read_csv(
                io.StringIO("\n".join((header, table))),
                dtype=dtype,
                index_col=None,
            )
        return df, lens
    except pd.errors.EmptyDataError:
        return None, lens
