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

import pandas as pd
from flask import Response, make_response, jsonify

from bentoml.handlers.base_handlers import BentoHandler, get_output_str
from bentoml.utils import is_url
from bentoml.utils.s3 import is_s3_url


def check_dataframe_column_contains(required_column_names, df):
    df_columns = set(df.columns)
    for col in required_column_names:
        if col not in df_columns:
            raise ValueError(
                "Missing columns: {}".format(
                    ",".join(set(required_column_names) - df_columns)
                )
            )


class DataframeHandler(BentoHandler):
    """Dataframe handler expects inputs from rest request or cli options that
     can be converted into a pandas Dataframe, and pass down the dataframe
     to user defined API function. It also returns response for REST API call
     or print result for CLI call

    Args:
        orient (string): Incoming json orient format for reading json data. Default is
            records.
        output_orient (string): Prefer json orient format for output result. Default is
            records.
        typ (string): Type of object to recover for read json with pandas. Default is
            frame
        input_columns ([string]): A list of column names that input data need to have.

    Raises:
        ValueError: Incoming data is missing required columns in input_columns
        ValueError: Incoming data format is not handled. Only json and csv
    """

    def __init__(
        self, orient="records", output_orient="records", typ="frame", input_columns=None
    ):
        self.orient = orient
        self.output_orient = output_orient or orient
        self.typ = typ
        self.input_columns = input_columns

    def handle_request(self, request, func):
        orient = request.headers.get("orient", self.orient)
        output_orient = request.headers.get("output_orient", self.output_orient)

        if request.content_type == "application/json":
            df = pd.read_json(
                request.data.decode("utf-8"), orient=orient, typ=self.typ, dtype=False
            )
        elif request.content_type == "text/csv":
            df = pd.read_csv(request.data.decode("utf-8"))
        else:
            return make_response(
                jsonify(
                    message="Request content-type not supported, only application/json "
                    "and text/csv are supported"
                ),
                400,
            )

        if self.typ == "frame" and self.input_columns is not None:
            check_dataframe_column_contains(self.input_columns, df)

        result = func(df)
        result = get_output_str(
            result, request.headers.get("output", "json"), output_orient
        )
        return Response(response=result, status=200, mimetype="application/json")

    def handle_cli(self, args, func):
        parser = argparse.ArgumentParser()
        parser.add_argument("--input", required=True)
        parser.add_argument(
            "-o", "--output", default="str", choices=["str", "json", "yaml"]
        )
        parser.add_argument("--orient", default=self.orient)
        parser.add_argument("--output_orient", default=self.output_orient)
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
                raise ValueError(
                    "Input file format not supported, BentoML cli only accepts .json "
                    "and .csv file"
                )
        else:
            # Assuming input string is JSON format
            try:
                df = pd.read_json(cli_input, orient=orient, typ=self.typ, dtype=False)
            except ValueError as e:
                raise ValueError(
                    "Unexpected input format, BentoML DataframeHandler expects json "
                    "string as input: {}".format(e)
                )

        if self.typ == "frame" and self.input_columns is not None:
            check_dataframe_column_contains(self.input_columns, df)

        result = func(df)
        result = get_output_str(result, parsed_args.output, output_orient)
        print(result)

    def handle_aws_lambda_event(self, event, func):
        orient = event["headers"].get("orient", self.orient)
        output_orient = event["headers"].get("output_orient", self.output_orient)

        if event["headers"]["Content-Type"] == "application/json":
            df = pd.read_json(event["body"], orient=orient, typ=self.typ, dtype=False)
        elif event["headers"]["Content-Type"] == "text/csv":
            df = pd.read_csv(event["body"])
        else:
            return {
                "statusCode": 400,
                "body": "Request content-type not supported, only application/json and "
                "text/csv are supported",
            }

        if self.typ == "frame" and self.input_columns is not None:
            check_dataframe_column_contains(self.input_columns, df)

        result = func(df)
        result = get_output_str(
            result, event["headers"].get("output", "json"), output_orient
        )
        return {"statusCode": 200, "body": result}
