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

from typing import Iterable, List
import sys
import json
import collections
import itertools
from io import StringIO

try:
    import pandas as pd
except ImportError:
    pd = None

from bentoml.utils import catch_exceptions
from bentoml.exceptions import BadInput, MissingDependencyException


class DataFrameState(object):
    def __init__(self, columns: List[str] = None, line_num: int = 0):
        self.columns = columns or []
        self.line_num = line_num


def check_dataframe_column_contains(required_column_names, df):
    df_columns = set(map(str, df.columns))
    for col in required_column_names:
        if col not in df_columns:
            raise BadInput(
                "Missing columns: {}, required_column:{}".format(
                    ",".join(set(required_column_names) - df_columns), df_columns
                )
            )


def _to_str(v):
    if v is None:
        return ""
    return str(v)


def _from_json_records(state: DataFrameState, table: list):
    if not state.line_num:  # make header
        state.columns = table[0].keys()
        yield itertools.chain(('',), state.columns)

    for tr in table:
        tds = (tr[c] for c in state.columns) if state.columns else tr.values()
        state.line_num += 1
        yield itertools.chain((state.line_num - 1,), tds)


def _from_json_values(state: DataFrameState, table: list):
    if not state.line_num:  # make header
        yield itertools.chain(('',), range(len(table[0])))

    for tr in table:
        state.line_num += 1
        yield itertools.chain((state.line_num - 1,), tr)


def _from_json_columns(state: DataFrameState, table: dict):
    if not state.line_num:  # make header
        state.columns = table.keys()
        yield itertools.chain(('',), state.columns)

    for row in next(iter(table.values())):
        if state.columns:
            tr = (table[col][row] for col in state.columns)
        else:
            tr = (table[col][row] for col in table.keys())
        state.line_num += 1
        yield itertools.chain((state.line_num - 1,), tr)


def _from_json_index(state: DataFrameState, table: dict):
    if not state.line_num:  # make header
        state.columns = next(iter(table.values())).keys()
        yield itertools.chain(('',), state.columns)

    for row in table.keys():
        if state.columns:
            tr = (table[row][col] for col in state.columns)
        else:
            tr = (td for td in table[row].values())
        state.line_num += 1
        yield itertools.chain((state.line_num - 1,), tr)


def _from_json_split(state: DataFrameState, table: dict):
    if not state.line_num:  # make header
        state.columns = table['columns']
        yield itertools.chain(('',), state.columns)

    if state.columns:
        _id_map = {k: i for i, k in enumerate(state.columns)}
        idxs = [_id_map[k] for k in table['columns']]
    for row in table['data']:
        if state.columns:
            tr = (row[idx] for idx in idxs)
        else:
            tr = row
        state.line_num += 1
        yield itertools.chain((state.line_num - 1,), tr)


def _from_csv_with_index(state: DataFrameState, table: List[str]):
    if not state.line_num:
        state.columns = table[0].split(',')[1:]
        yield table[0]
    for row_str in table[1:]:
        if not row_str.strip():  # skip blank line
            continue
        state.line_num += 1
        yield f"{str(state.line_num - 1)},{row_str.split(',', maxsplit=1)[1]}"


def _from_csv_without_index(state: DataFrameState, table: List[str]):
    if not state.line_num:
        state.columns = table[0].split(',')
        yield "," + table[0]
    for row_str in table[1:]:
        if not row_str.strip():  # skip blank line
            continue
        state.line_num += 1
        yield f"{str(state.line_num - 1)},{row_str.strip()}"


def _detect_orient(table):
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
def _guess_orient(table):
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


_ORIENT_MAP = {
    'records': _from_json_records,
    'columns': _from_json_columns,
    'values': _from_json_values,
    'split': _from_json_split,
    'index': _from_json_index,
    # 'table': _from_json_table,
}


def _dataframe_csv_from_input(tables, content_types, orients):
    state = DataFrameState()
    for table_id, (table, content_type, orient) in enumerate(
        zip(tables, content_types, orients)
    ):
        content_type = content_type or "application/json"
        if content_type.lower() == "application/json":
            # Keep order when loading data
            if sys.version_info >= (3, 6):
                table = json.loads(table.decode('utf-8'))
            else:
                table = json.loads(
                    table.decode('utf-8'), object_pairs_hook=collections.OrderedDict
                )
        elif content_type.lower() == "text/csv":
            table = table.decode('utf-8').split('\n')
            if not table:
                continue
        else:
            raise BadInput(f'Invalid content_type for DataframeInput: {content_type}')

        if content_type.lower() == "application/json":
            if not orient:
                orient = _detect_orient(table)

            if not orient:
                raise BadInput(
                    'Unable to detect Json orient, please specify the format orient.'
                )

            if orient not in _ORIENT_MAP:
                raise NotImplementedError(
                    f'Json orient "{orient}" is not supported now'
                )

            _from_json = _ORIENT_MAP[orient]

            try:
                for line in _from_json(state, table):
                    yield line, table_id if state.line_num else None
            except Exception as e:  # pylint:disable=broad-except
                guessed_orient = _guess_orient(table)
                if guessed_orient:
                    raise BadInput(
                        f'Not a valid "{orient}" oriented Json. '
                        f'The orient seems to be "{guessed_orient}". '
                        f'Try DataframeInput(orient="{guessed_orient}") instead.'
                    ) from e
                else:
                    raise BadInput(f'Not a valid "{orient}" oriented Json. ') from e

            continue
        elif content_type.lower() == "text/csv":
            if table[0].strip().startswith(','):  # csv with index column
                for line in _from_csv_with_index(state, table):
                    yield line, table_id if state.line_num else None
            else:
                for line in _from_csv_without_index(state, table):
                    yield line, table_id if state.line_num else None


def _gen_slice(ids):
    start = -1
    i = -1
    for i, id_ in enumerate(ids):
        if start == -1:
            start = i
            continue

        if ids[start] != id_:
            yield slice(start, i)
            start = i
            continue
    yield slice(start, i + 1)


def read_dataframes_from_json_n_csv(
    datas: Iterable["pd.DataFrame"], content_types: Iterable[str], orient: str = None,
) -> ("pd.DataFrame", Iterable[slice]):
    '''
    load detaframes from multiple raw datas in json or csv fromat, efficiently

    Background: Each calling of pandas.read_csv or pandas.read_json cost about 100ms,
    no matter how many lines it contains. Concat jsons/csvs before read_json/read_csv
    to improve performance.
    '''
    if not pd:
        raise MissingDependencyException('pandas required')
    try:
        rows_csv_with_id = [
            (tds if isinstance(tds, str) else ','.join(map(_to_str, tds)), table_id)
            for tds, table_id in _dataframe_csv_from_input(
                datas, content_types, itertools.repeat(orient)
            )
            if tds is not None
        ]
    except (TypeError, ValueError) as e:
        raise BadInput('Invalid input format for DataframeInput') from e

    str_csv = [r for r, _ in rows_csv_with_id]
    df_str_csv = '\n'.join(str_csv)
    df_merged = pd.read_csv(StringIO(df_str_csv), index_col=0)

    dfs_id = [i for _, i in rows_csv_with_id][1:]
    slices = _gen_slice(dfs_id)
    return df_merged, slices
