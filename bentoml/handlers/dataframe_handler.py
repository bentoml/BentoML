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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import argparse
from io import StringIO

import pandas as pd
from flask import Response

from bentoml.handlers.base_handlers import (
    BentoHandler,
    api_func_result_to_json,
    PANDAS_DATAFRAME_TO_DICT_ORIENT_OPTIONS,
)
from bentoml.utils import is_url
from bentoml.utils.s3 import is_s3_url
from bentoml.exceptions import BadInput


def _check_dataframe_column_contains(required_column_names, df):
    df_columns = set(map(str, df.columns))
    for col in required_column_names:
        if col not in df_columns:
            raise BadInput(
                "Missing columns: {}, required_column:{}".format(
                    ",".join(set(required_column_names) - df_columns), df_columns
                )
            )


class DataframeHandler(BentoHandler):
    """Dataframe handler expects inputs from HTTP request or cli arguments that
     can be converted into a pandas Dataframe. It passes down the dataframe
     to user defined API function and returns response for REST API call
     or print result for CLI call

    Args:
        orient (str): Incoming json orient format for reading json data. Default is
            records.
        output_orient (str): Prefer json orient format for output result. Default is
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

    def __init__(
        self, orient="records", output_orient="records", typ="frame", input_dtypes=None
    ):
        self.orient = orient
        self.output_orient = output_orient or orient

        assert self.orient in PANDAS_DATAFRAME_TO_DICT_ORIENT_OPTIONS, (
            f"Invalid option 'orient'='{self.orient}', valid options are "
            f"{PANDAS_DATAFRAME_TO_DICT_ORIENT_OPTIONS}"
        )
        assert self.orient in PANDAS_DATAFRAME_TO_DICT_ORIENT_OPTIONS, (
            f"Invalid 'output_orient'='{self.orient}', valid options are "
            f"{PANDAS_DATAFRAME_TO_DICT_ORIENT_OPTIONS}"
        )

        self.typ = typ
        self.input_dtypes = input_dtypes

        if isinstance(self.input_dtypes, list):
            self.input_dtypes = dict(
                (str(index), dtype) for index, dtype in enumerate(self.input_dtypes)
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
        if request.content_type == "application/json":
            df = pd.read_json(
                request.data.decode("utf-8"),
                orient=self.orient,
                typ=self.typ,
                dtype=False,
            )
        elif request.content_type == "text/csv":
            csv_string = StringIO(request.data.decode('utf-8'))
            df = pd.read_csv(csv_string)
        else:
            raise BadInput(
                "Request content-type not supported, only application/json and "
                "text/csv are supported"
            )

        if self.typ == "frame" and self.input_dtypes is not None:
            _check_dataframe_column_contains(self.input_dtypes, df)

        result = func(df)
        json_output = api_func_result_to_json(
            result, pandas_dataframe_orient=self.output_orient
        )
        return Response(response=json_output, status=200, mimetype="application/json")

    def handle_cli(self, args, func):
        parser = argparse.ArgumentParser()
        parser.add_argument("--input", required=True)
        parser.add_argument("-o", "--output", default="str", choices=["str", "json"])
        parser.add_argument(
            "--orient",
            default=self.orient,
            choices=PANDAS_DATAFRAME_TO_DICT_ORIENT_OPTIONS,
        )
        parser.add_argument(
            "--output_orient",
            default=self.output_orient,
            choices=PANDAS_DATAFRAME_TO_DICT_ORIENT_OPTIONS,
        )
        parsed_args = parser.parse_args(args)

        orient = parsed_args.orient
        output_orient = parsed_args.output_orient
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
                    "Unexpected input format, BentoML DataframeHandler expects json "
                    "string as input: {}".format(e)
                )

        if self.typ == "frame" and self.input_dtypes is not None:
            _check_dataframe_column_contains(self.input_dtypes, df)

        result = func(df)
        if parsed_args.output == 'json':
            result = api_func_result_to_json(
                result, pandas_dataframe_orient=output_orient
            )
        else:
            result = str(result)
        print(result)

    def handle_aws_lambda_event(self, event, func):
        if event["headers"]["Content-Type"] == "application/json":
            df = pd.read_json(
                event["body"], orient=self.orient, typ=self.typ, dtype=False
            )
        elif event["headers"]["Content-Type"] == "text/csv":
            df = pd.read_csv(event["body"])
        else:
            raise BadInput(
                "Request content-type not supported, only application/json and "
                "text/csv are supported"
            )

        if self.typ == "frame" and self.input_dtypes is not None:
            _check_dataframe_column_contains(self.input_dtypes, df)

        result = func(df)
        result = api_func_result_to_json(
            result, pandas_dataframe_orient=self.output_orient
        )
        return {"statusCode": 200, "body": result}
