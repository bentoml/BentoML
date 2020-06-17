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

from typing import Iterable
import os
import sys
import json
import collections
import itertools
import argparse
from io import StringIO

try:
    import pandas as pd
except ImportError:
    pd = None

from bentoml.adapters.base_input import (
    BaseInputAdapter,
    PANDAS_DATAFRAME_TO_DICT_ORIENT_OPTIONS,
)
from bentoml.utils import is_url
from bentoml.marshal.utils import SimpleResponse, SimpleRequest
from bentoml.utils.s3 import is_s3_url
from bentoml.exceptions import BadInput, MissingDependencyException


def _check_dataframe_column_contains(required_column_names, df):
    df_columns = set(map(str, df.columns))
    for col in required_column_names:
        if col not in df_columns:
            raise BadInput(
                "Missing columns: {}, required_column:{}".format(
                    ",".join(set(required_column_names) - df_columns), df_columns
                )
            )


def _to_csv_cell(v):
    if v is None:
        return ""
    return str(v)


def _dataframe_csv_from_input(raws, content_types):
    row_sum = -1
    columns = []
    for i, (data, content_type) in enumerate(zip(raws, content_types)):
        if not content_type or content_type.lower() == "application/json":
            # load json with order preserved
            if sys.version_info >= (3, 6):
                od = json.loads(data.decode('utf-8'))
            else:
                od = json.loads(
                    data.decode('utf-8'), object_pairs_hook=collections.OrderedDict
                )

            if isinstance(od, list):
                if isinstance(od[0], dict):  # orient: records
                    if row_sum == -1:  # make header
                        columns = od[0].keys()
                        yield ",".join(
                            itertools.chain(('',), map(_to_csv_cell, columns))
                        ), None
                        row_sum += 1

                    for datas_row in od:
                        datas = (
                            (datas_row[c] for c in columns)
                            if columns
                            else datas_row.values()
                        )
                        yield ','.join(
                            itertools.chain((str(row_sum),), map(_to_csv_cell, datas))
                        ), i
                        row_sum += 1
                else:  # orient: values
                    if row_sum == -1:  # make header
                        yield ",".join(
                            itertools.chain(('',), map(_to_csv_cell, range(len(od[0]))))
                        ), None
                        row_sum += 1

                    for datas_row in od:
                        yield ','.join(
                            itertools.chain(
                                (str(row_sum),), map(_to_csv_cell, datas_row)
                            )
                        ), i
                        row_sum += 1
            elif isinstance(od, dict):
                if isinstance(next(iter(od.values())), dict):  # orient: columns
                    if row_sum == -1:  # make header
                        columns = od.keys()
                        yield ",".join(
                            itertools.chain(('',), map(_to_csv_cell, columns))
                        ), None
                        row_sum += 1

                    for row in next(iter(od.values())):
                        if columns:
                            datas_row = (od[col][row] for col in columns)
                        else:
                            datas_row = (od[col][row] for col in od.keys())
                        yield ','.join(
                            itertools.chain(
                                (str(row_sum),), map(_to_csv_cell, datas_row)
                            )
                        ), i
                        row_sum += 1
                else:
                    raise NotImplementedError()

        elif content_type.lower() == "text/csv":
            data_str = data.decode('utf-8')
            row_strs = data_str.split('\n')
            if not row_strs:
                continue
            if row_strs[0].strip().startswith(','):  # csv with index column
                if row_sum == -1:
                    columns = row_strs[0].strip().split(',')[1:]
                    yield row_strs[0].strip(), None
                for row_str in row_strs[1:]:
                    if not row_str.strip():  # skip blank line
                        continue
                    yield f"{str(row_sum)},{row_str.split(',', maxsplit=1)[1]}", i
                    row_sum += 1
            else:
                if row_sum == -1:
                    columns = row_strs[0].strip().split(',')
                    yield "," + row_strs[0].strip(), None
                for row_str in row_strs[1:]:
                    if not row_str.strip():  # skip blank line
                        continue
                    yield f"{str(row_sum)},{row_str.strip()}", i
                    row_sum += 1
        else:
            raise BadInput(f'Invalid content_type for DataframeInput: {content_type}')


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
    datas: Iterable["pd.DataFrame"], content_types: Iterable[str]
) -> ("pd.DataFrame", Iterable[slice]):
    '''
    load detaframes from multiple raw datas in json or csv fromat, efficiently

    Background: Each calling of pandas.read_csv or pandas.read_json cost about 100ms,
    no matter how many lines it contains. Concat jsons/csvs before read_json/read_csv
    to improve performance.
    '''
    try:
        rows_csv_with_id = [r for r in _dataframe_csv_from_input(datas, content_types)]
    except TypeError:
        raise BadInput('Invalid input format for DataframeInput') from None

    str_csv = [r for r, _ in rows_csv_with_id]
    df_str_csv = '\n'.join(str_csv)
    df_merged = pd.read_csv(StringIO(df_str_csv), index_col=0)

    dfs_id = [i for _, i in rows_csv_with_id][1:]
    slices = _gen_slice(dfs_id)
    return df_merged, slices


class DataframeInput(BaseInputAdapter):
    """DataframeInput expects inputs from HTTP request or cli arguments that
        can be converted into a pandas Dataframe. It passes down the dataframe
        to user defined API function and returns response for REST API call
        or print result for CLI call

    Args:
        orient (str): Incoming json orient format for reading json data. Default is
            records.
        typ (str): Type of object to recover for read json with pandas. Default is
            frame
        input_dtypes ({str:str}): describing expected input data types of the input
            dataframe, it must be either a dict of column name and data type, or a list
            of data types listed by column index in the dataframe

    Raises:
        ValueError: Incoming data is missing required columns in input_dtypes
        ValueError: Incoming data format can not be handled. Only json and csv
    """

    BATCH_MODE_SUPPORTED = True

    def __init__(
        self,
        orient="records",
        typ="frame",
        input_dtypes=None,
        is_batch_input=True,
        **base_kwargs,
    ):
        if not is_batch_input:
            raise ValueError('DataframeInput can not accpept none batch inputs')
        super(DataframeInput, self).__init__(
            is_batch_input=is_batch_input, **base_kwargs
        )

        # Verify pandas imported properly and retry import if it has failed initially
        global pd  # pylint: disable=global-statement
        if pd is None:
            try:
                import pandas as pd  # pylint: disable=redefined-outer-name
            except ImportError:
                raise MissingDependencyException(
                    "Missing required dependency 'pandas' for DataframeInput, install"
                    "with `pip install pandas`"
                )

        self.orient = orient

        assert self.orient in PANDAS_DATAFRAME_TO_DICT_ORIENT_OPTIONS, (
            f"Invalid option 'orient'='{self.orient}', valid options are "
            f"{PANDAS_DATAFRAME_TO_DICT_ORIENT_OPTIONS}"
        )
        self.typ = typ
        self.input_dtypes = input_dtypes

        if isinstance(self.input_dtypes, list):
            self.input_dtypes = dict(
                (str(index), dtype) for index, dtype in enumerate(self.input_dtypes)
            )

    @property
    def pip_dependencies(self):
        return ['pandas']

    @property
    def config(self):
        base_config = super(DataframeInput, self).config
        return dict(
            base_config,
            orient=self.orient,
            typ=self.typ,
            input_dtypes=self.input_dtypes,
        )

    def _get_type(self, item):
        if item.startswith('int'):
            return 'integer'
        if item.startswith('float') or item.startswith('double'):
            return 'number'
        if item.startswith('str') or item.startswith('date'):
            return 'string'
        if item.startswith('bool'):
            return 'boolean'
        return 'object'

    @property
    def request_schema(self):
        default = {"application/json": {"schema": {"type": "object"}}}
        if self.input_dtypes is None:
            return default

        if isinstance(self.input_dtypes, dict):
            return {
                "application/json": {  # For now, only declare JSON on docs.
                    "schema": {
                        "type": "object",
                        "properties": {
                            k: {"type": "array", "items": {"type": self._get_type(v)}}
                            for k, v in self.input_dtypes.items()
                        },
                    }
                }
            }

        return default

    def handle_request(self, request, func):
        if request.content_type == "text/csv":
            csv_string = StringIO(request.data.decode('utf-8'))
            df = pd.read_csv(csv_string)
        else:
            # Optimistically assuming Content-Type to be "application/json"
            try:
                df = pd.read_json(
                    request.data.decode("utf-8"),
                    orient=self.orient,
                    typ=self.typ,
                    dtype=False,
                )
            except ValueError:
                raise BadInput(
                    "Failed parsing request data, only Content-Type application/json "
                    "and text/csv are supported in BentoML DataframeInput"
                )

        if self.typ == "frame" and self.input_dtypes is not None:
            _check_dataframe_column_contains(self.input_dtypes, df)

        result = func(df)
        return self.output_adapter.to_response(result, request)

    def handle_batch_request(
        self, requests: Iterable[SimpleRequest], func
    ) -> Iterable[SimpleResponse]:

        datas = [r.data for r in requests]
        content_types = [
            r.formated_headers.get('content-type', 'application/json') for r in requests
        ]
        # TODO: check content_type

        df_conc, slices_generator = read_dataframes_from_json_n_csv(
            datas, content_types
        )
        slices = list(slices_generator)

        result_conc = func(df_conc)

        return self.output_adapter.to_batch_response(
            result_conc, slices=slices, requests=requests,
        )

    def handle_cli(self, args, func):
        parser = argparse.ArgumentParser()
        parser.add_argument("--input", required=True)
        parser.add_argument(
            "--orient",
            default=self.orient,
            choices=PANDAS_DATAFRAME_TO_DICT_ORIENT_OPTIONS,
        )
        parsed_args, unknown_args = parser.parse_known_args(args)

        orient = parsed_args.orient
        cli_input = parsed_args.input

        if os.path.isfile(cli_input) or is_s3_url(cli_input) or is_url(cli_input):
            if cli_input.endswith(".csv"):
                df = pd.read_csv(cli_input)
            elif cli_input.endswith(".json"):
                df = pd.read_json(cli_input, orient=orient, typ=self.typ, dtype=False)
            else:
                raise BadInput(
                    "Input file format not supported, BentoML cli only accepts .json "
                    "and .csv file"
                )
        else:
            # Assuming input string is JSON format
            try:
                df = pd.read_json(cli_input, orient=orient, typ=self.typ, dtype=False)
            except ValueError as e:
                raise BadInput(
                    "Unexpected input format, BentoML DataframeInput expects json "
                    "string as input: {}".format(e)
                )

        if self.typ == "frame" and self.input_dtypes is not None:
            _check_dataframe_column_contains(self.input_dtypes, df)

        result = func(df)
        self.output_adapter.to_cli(result, unknown_args)

    def handle_aws_lambda_event(self, event, func):
        if event["headers"].get("Content-Type", None) == "text/csv":
            df = pd.read_csv(event["body"])
        else:
            # Optimistically assuming Content-Type to be "application/json"
            try:
                df = pd.read_json(
                    event["body"], orient=self.orient, typ=self.typ, dtype=False
                )
            except ValueError:
                raise BadInput(
                    "Failed parsing request data, only Content-Type application/json "
                    "and text/csv are supported in BentoML DataframeInput"
                )

        if self.typ == "frame" and self.input_dtypes is not None:
            _check_dataframe_column_contains(self.input_dtypes, df)

        result = func(df)
        return self.output_adapter.to_aws_lambda_event(result, event)
