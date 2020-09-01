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
from typing import BinaryIO, Iterable, Tuple

from bentoml.adapters.file_input import FileInput
from bentoml.adapters.utils import check_file_extension
from bentoml.exceptions import MissingDependencyException
from bentoml.types import InferenceTask
from bentoml.utils.dataframe_util import (
    PANDAS_DATAFRAME_TO_JSON_ORIENT_OPTIONS,
    read_dataframes_from_json_n_csv,
)
from bentoml.utils.lazy_loader import LazyLoader

pandas = LazyLoader('pandas', globals(), 'pandas')

DataFrameTask = InferenceTask[BinaryIO]
ApiFuncArgs = Tuple['pandas.DataFrame']


class DataframeInput(FileInput):
    def __init__(
        self, orient=None, typ="frame", columns=None, input_dtypes=None, **base_kwargs,
    ):
        super().__init__(**base_kwargs)

        # Verify pandas imported properly and retry import if it has failed initially
        if pandas is None:
            raise MissingDependencyException(
                "Missing required dependency 'pandas' for DataframeInput, install "
                "with `pip install pandas`"
            )
        if typ != "frame":
            raise NotImplementedError()
        assert not orient or orient in PANDAS_DATAFRAME_TO_JSON_ORIENT_OPTIONS, (
            f"Invalid option 'orient'='{orient}', valid options are "
            f"{PANDAS_DATAFRAME_TO_JSON_ORIENT_OPTIONS}"
        )

        assert (
            columns is None or input_dtypes is None or set(input_dtypes) == set(columns)
        ), "input_dtypes must match columns"

        self.typ = typ
        self.orient = orient
        self.columns = columns
        if isinstance(input_dtypes, (list, tuple)):
            self.input_dtypes = dict(
                (index, dtype) for index, dtype in enumerate(input_dtypes)
            )
        else:
            self.input_dtypes = input_dtypes

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

    @classmethod
    def _detect_format(cls, task: InferenceTask) -> str:
        if task.context.http_headers.content_type == "application/json":
            return "json"
        if task.context.http_headers.content_type == "text/csv":
            return "csv"
        file_name = getattr(task.data, "name", "")
        if check_file_extension(file_name, (".csv",)):
            return "csv"
        return "json"

    def extract_user_func_args(
        self, tasks: Iterable[InferenceTask[bytes]]
    ) -> ApiFuncArgs:
        fmts = (self._detect_format(task) for task in tasks)
        datas = (task.data.read() for task in tasks)

        df_csv, batchs = read_dataframes_from_json_n_csv(
            datas, fmts, orient=self.orient, columns=self.columns
        )
        for task, batch in zip(tasks, batchs):
            if batch:
                task.batch = batch
            else:
                task.discard(
                    http_status=400,
                    err_msg=f"{self.__class__.__name__}Wrong input format.",
                )
        df = pandas.read_csv(
            io.StringIO(df_csv, index_col=None, dtype=self.input_dtypes)
        )
        return (df,)


'''
from typing import Iterable
import os
import argparse
from io import StringIO
from typing import Iterable

try:
    import pandas as pd
except ImportError:
    pd = None
import flask

from bentoml.adapters.base_input import BaseInputAdapter
from bentoml.utils import is_url
from bentoml.utils.dataframe_util import (
    read_dataframes_from_json_n_csv,
    check_dataframe_column_contains,
    PANDAS_DATAFRAME_TO_JSON_ORIENT_OPTIONS,
)
from bentoml.types import HTTPRequest, HTTPResponse
from bentoml.utils.s3 import is_s3_url
from bentoml.exceptions import BadInput, MissingDependencyException


class DataframeInput(BaseInputAdapter):
    """DataframeInput expects inputs from HTTP request or cli arguments that
        can be converted into a pandas Dataframe. It passes down the dataframe
        to user defined API function and returns response for REST API call
        or print result for CLI call

    Args:
        orient (str or None): Incoming json orient format for reading json data.
            Default is None which means automatically detect.
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
        orient=None,
        typ="frame",
        input_dtypes=None,
        is_batch_input=True,
        **base_kwargs,
    ):
        if not is_batch_input:
            raise ValueError('DataframeInput can not accept none batch inputs')
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
                    "Missing required dependency 'pandas' for DataframeInput, install "
                    "with `pip install pandas`"
                )

        self.orient = orient

        assert (
            not self.orient or self.orient in PANDAS_DATAFRAME_TO_JSON_ORIENT_OPTIONS
        ), (
            f"Invalid option 'orient'='{self.orient}', valid options are "
            f"{PANDAS_DATAFRAME_TO_JSON_ORIENT_OPTIONS}"
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

    def handle_request(self, request: flask.Request):
        if request.content_type == "text/csv":
            csv_string = StringIO(request.get_data(as_text=True))
            df = pd.read_csv(csv_string)
        else:
            # Optimistically assuming Content-Type to be "application/json"
            try:
                df = pd.read_json(
                    request.get_data(as_text=True), orient=self.orient, typ=self.typ,
                )
            except ValueError:
                raise BadInput(
                    "Failed parsing request data, only Content-Type application/json "
                    "and text/csv are supported in BentoML DataframeInput"
                )

        if self.typ == "frame" and self.input_dtypes is not None:
            check_dataframe_column_contains(self.input_dtypes, df)

        result = func(df)
        return self.output_adapter.to_response(result, request)

    def handle_batch_request(
        self, requests: Iterable[HTTPRequest], func
    ) -> Iterable[HTTPResponse]:

        datas = [r.body for r in requests]
        content_types = [
            r.parsed_headers.content_type or 'application/json' for r in requests
        ]

        df_conc, slices_generator = read_dataframes_from_json_n_csv(
            datas, content_types, self.orient
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
            choices=PANDAS_DATAFRAME_TO_JSON_ORIENT_OPTIONS,
        )
        parsed_args, unknown_args = parser.parse_known_args(args)

        orient = parsed_args.orient
        cli_input = parsed_args.input

        if os.path.isfile(cli_input) or is_s3_url(cli_input) or is_url(cli_input):
            if cli_input.endswith(".csv"):
                df = pd.read_csv(cli_input)
            elif cli_input.endswith(".json"):
                df = pd.read_json(cli_input, orient=orient, typ=self.typ)
            else:
                raise BadInput(
                    "Input file format not supported, BentoML cli only accepts .json "
                    "and .csv file"
                )
        else:
            # Assuming input string is JSON format
            try:
                df = pd.read_json(cli_input, orient=orient, typ=self.typ)
            except ValueError as e:
                raise BadInput(
                    "Unexpected input format, BentoML DataframeInput expects json "
                    "string as input: {}".format(e)
                )

        if self.typ == "frame" and self.input_dtypes is not None:
            check_dataframe_column_contains(self.input_dtypes, df)

        result = func(df)
        self.output_adapter.to_cli(result, unknown_args)

    def handle_aws_lambda_event(self, event, func):
        if event["headers"].get("Content-Type", None) == "text/csv":
            df = pd.read_csv(event["body"])
        else:
            # Optimistically assuming Content-Type to be "application/json"
            try:
                df = pd.read_json(event["body"], orient=self.orient, typ=self.typ)
            except ValueError:
                raise BadInput(
                    "Failed parsing request data, only Content-Type application/json "
                    "and text/csv are supported in BentoML DataframeInput"
                )

        if self.typ == "frame" and self.input_dtypes is not None:
            check_dataframe_column_contains(self.input_dtypes, df)

        result = func(df)
        return self.output_adapter.to_aws_lambda_event(result, event)
'''
