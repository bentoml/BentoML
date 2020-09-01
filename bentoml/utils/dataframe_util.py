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

from typing import Iterable, Iterator, Mapping
import sys
import json
import collections
import itertools

from bentoml.utils import catch_exceptions
from bentoml.utils.csv import csv_split, csv_splitline, csv_quote, csv_unquote, csv_row
from bentoml.exceptions import BadInput, MissingDependencyException


def check_dataframe_column_contains(required_column_names, df):
    df_columns = set(map(str, df.columns))
    for col in required_column_names:
        if col not in df_columns:
            raise BadInput(
                "Missing columns: {}, required_column:{}".format(
                    ",".join(set(required_column_names) - df_columns), df_columns
                )
            )


def detect_orient(table):
    if isinstance(table, list):
        if isinstance(table[0], dict):
            return 'records'
        else:
            return 'values'
    elif isinstance(table, dict):
        if isinstance(next(iter(table.values())), dict):
            return 'columns'
    # Do not need more auto orients supports than official pandas
    return None


@catch_exceptions(Exception, fallback=None)
def guess_orient(table):
    if isinstance(table, list):
        if isinstance(table[0], dict):
            return 'records'
        else:
            return 'values'
    elif isinstance(table, dict):
        if all(isinstance(v, list) for v in iter(table.values())):
            if 'columns' in table and 'index' in table:
                return 'split'
            return None
        if all(isinstance(v, dict) for v in iter(table.values())):
            if any(isinstance(k, str) and not k.isnumeric() for k in table.keys()):
                return 'columns'
            if any(
                isinstance(k, str) and not k.isnumeric()
                for k in next(iter(table.values())).keys()
            ):
                return 'index'
            return {'columns', 'index'}
        if 'schema' in table and 'data' in table and 'primaryKey' in table['schema']:
            return 'table'
        return None


class DataFrameState(object):
    def __init__(self, columns: Mapping[str, int] = None):
        self.columns = columns


def _from_json_records(state: DataFrameState, table: list):
    if state.columns is None:  # make header
        state.columns = {k: i for i, k in enumerate(table[0].keys())}
        for tr in table:
            yield csv_row(tr.values())
    else:
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


def _dataframe_csv_from_input(table, fmt, orient, state):
    fmt = fmt or "json"
    if fmt == "json":
        # Keep order when loading data
        if sys.version_info >= (3, 6):
            table = json.loads(table.decode('utf-8'))
        else:
            table = json.loads(
                table.decode('utf-8'), object_pairs_hook=collections.OrderedDict
            )
    elif fmt == "csv":
        table = csv_splitline(table.decode('utf-8'))
        if not table:
            return tuple()
    else:
        raise BadInput(f'Invalid format for DataframeInput: {fmt}')

    if fmt == "json":
        if not orient:
            orient = detect_orient(table)
        if not orient:
            raise BadInput(
                'Unable to detect Json orient, please specify the format orient.'
            )
        if orient not in _ORIENT_MAP:
            raise NotImplementedError(f'Json orient "{orient}" is not supported now')
        _from_json = _ORIENT_MAP[orient]
        try:
            return tuple(_from_json(state, table))
        except Exception as e:  # pylint:disable=broad-except
            guessed_orient = guess_orient(table)
            if guessed_orient:
                raise BadInput(
                    f'Not a valid "{orient}" oriented Json. '
                    f'The orient seems to be "{guessed_orient}". '
                    f'Try DataframeInput(orient="{guessed_orient}") instead.'
                ) from e
            else:
                raise BadInput(f'Not a valid "{orient}" oriented Json. ') from e
            return tuple()
    elif fmt == "csv":
        return tuple(_from_csv_without_index(state, table))


def read_dataframes_from_json_n_csv(
    datas: Iterable[bytes], formats: Iterable[str], orient: str = None, columns=None
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
    lens = tuple(len(trs) for trs in trs_list)
    header = ",".join(csv_quote(td) for td in state.columns)
    table = header + "\n" + "\n".join((tr) for trs in trs_list for tr in trs)
    return table, lens
