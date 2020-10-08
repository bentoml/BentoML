# Copyright 2019 Atalaya Tech, Inc.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

# http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import io
import itertools
import json
from typing import Iterable, Iterator, Mapping

from bentoml.exceptions import BadInput
from bentoml.utils import catch_exceptions
from bentoml.utils.csv import (
    csv_quote,
    csv_row,
    csv_split,
    csv_splitlines,
    csv_unquote,
)
from bentoml.utils.lazy_loader import LazyLoader

pandas = LazyLoader('pandas', globals(), 'pandas')


def check_dataframe_column_contains(required_column_names, df):
    df_columns = set(map(str, df.columns))
    for col in required_column_names:
        if col not in df_columns:
            raise BadInput(
                "Missing columns: {}, required_column:{}".format(
                    ",".join(set(required_column_names) - df_columns), df_columns
                )
            )


@catch_exceptions(Exception, fallback=None)
def guess_orient(table, strict=False):
    if isinstance(table, list):
        if not table:
            if strict:
                return {"records", "values"}
            else:
                return "records"
        if isinstance(table[0], dict):
            return 'records'
        else:
            return 'values'
    elif isinstance(table, dict):
        if set(table) == {"columns", "index", "data"}:
            return 'split'
        if set(table) == {"schema", "data"} and "primaryKey" in table["schema"]:
            return 'table'
        if strict:
            return {'columns', "index"}
        else:
            return "columns"


class DataFrameState(object):
    def __init__(self, columns: Mapping[str, int] = None):
        self.columns = columns


def _from_json_records(state: DataFrameState, table: list):
    if state.columns is None:  # make header
        state.columns = {k: i for i, k in enumerate(table[0].keys())}
    for tr in table:
        yield csv_row(tr[c] for c in state.columns)


def _from_json_values(_: DataFrameState, table: list):
    for tr in table:
        yield csv_row(tr)


def _from_json_columns(state: DataFrameState, table: dict):
    if state.columns is None:  # make header
        state.columns = {k: i for i, k in enumerate(table.keys())}
    for row in next(iter(table.values())):
        yield csv_row(table[col][row] for col in state.columns)


def _from_json_index(state: DataFrameState, table: dict):
    if state.columns is None:  # make header
        state.columns = {k: i for i, k in enumerate(next(iter(table.values())).keys())}
        for row in table.keys():
            yield csv_row(td for td in table[row].values())
    else:
        for row in table.keys():
            yield csv_row(table[row][col] for col in state.columns)


def _from_json_split(state: DataFrameState, table: dict):
    table_columns = {k: i for i, k in enumerate(table['columns'])}

    if state.columns is None:  # make header
        state.columns = table_columns
        for row in table['data']:
            yield csv_row(row)
    else:
        idxs = [state.columns[k] for k in table_columns]
        for row in table['data']:
            yield csv_row(row[idx] for idx in idxs)


def _from_csv_without_index(state: DataFrameState, table: Iterator[str]):
    row_str = next(table)  # skip column names
    table_columns = tuple(csv_unquote(s) for s in csv_split(row_str, ','))

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
        idxs = [state.columns[k] for k in table_columns]
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


_ORIENT_MAP = {
    'records': _from_json_records,
    'columns': _from_json_columns,
    'values': _from_json_values,
    'split': _from_json_split,
    'index': _from_json_index,
    # 'table': _from_json_table,
}

PANDAS_DATAFRAME_TO_JSON_ORIENT_OPTIONS = {k for k in _ORIENT_MAP}


def _dataframe_csv_from_input(table: str, fmt, orient, state):
    try:
        if not fmt or fmt == "json":
            table = json.loads(table)
            if not orient:
                orient = guess_orient(table, strict=False)
            else:
                guessed_orient = guess_orient(table, strict=True)
                if orient != guessed_orient and orient not in guessed_orient:
                    print("xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx", orient)
                    return None
            if orient not in _ORIENT_MAP:
                return None
            _from_json = _ORIENT_MAP[orient]
            try:
                return tuple(_from_json(state, table))
            except (TypeError, AttributeError, KeyError, IndexError):
                return None
        elif fmt == "csv":
            table = csv_splitlines(table)
            return tuple(_from_csv_without_index(state, table))
        else:
            return None
    except (json.JSONDecodeError):
        return None


def read_dataframes_from_json_n_csv(
    datas: Iterable[str],
    formats: Iterable[str],
    orient: str = None,
    columns=None,
    dtype=None,
) -> ("pandas.DataFrame", Iterable[slice]):
    '''
    load dataframes from multiple raw datas in json or csv format, efficiently

    Background: Each calling of pandas.read_csv or pandas.read_json cost about 100ms,
    no matter how many lines it contains. Concat jsons/csvs before read_json/read_csv
    to improve performance.
    '''
    state = DataFrameState(
        columns={k: i for i, k in enumerate(columns)} if columns else None
    )
    trs_list = tuple(
        _dataframe_csv_from_input(t, fmt, orient, state)
        for t, fmt in zip(datas, formats)
    )
    header = ",".join(csv_quote(td) for td in state.columns) if state.columns else None
    lens = tuple(len(trs) if trs else 0 for trs in trs_list)
    table = "\n".join(tr for trs in trs_list if trs is not None for tr in trs)
    try:
        if not header:
            df = pandas.read_csv(
                io.StringIO(table), index_col=None, dtype=dtype, header=None,
            )
        else:
            df = pandas.read_csv(
                io.StringIO("\n".join((header, table))), index_col=None, dtype=dtype,
            )
        return df, lens
    except pandas.errors.EmptyDataError:
        return None, lens
