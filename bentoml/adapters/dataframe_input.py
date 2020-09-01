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

import argparse
from typing import Iterable, Tuple, Iterator

from bentoml.adapters.base_input import BaseInputAdapter
from bentoml.exceptions import MissingDependencyException
from bentoml.types import (
    HTTPRequest,
    InferenceTask,
    InferenceContext,
    AwsLambdaEvent,
    ApiFuncArgs,
)
from bentoml.utils.dataframe_util import (
    check_dataframe_column_contains,
    PANDAS_DATAFRAME_TO_JSON_ORIENT_OPTIONS,
)

try:
    import pandas as pd
except ImportError:
    raise MissingDependencyException(
        "Missing required dependency 'pandas' for DataframeInput, install "
        "with `pip install pandas`"
    )


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

    def from_cli(self, cli_args: Tuple[str, ...]) -> Iterator[InferenceTask]:
        parser = argparse.ArgumentParser()
        input_g = parser.add_mutually_exclusive_group(required=True)

        strinput = input_g.add_argument_group()
        strinput.add_argument("--input", nargs="+", type=str, required=True)
        typeinput = strinput.add_mutually_exclusive_group(required=True)
        typeinput.add_argument("--csv")
        typeinput.add_argument("--json")

        input_g.add_argument("--input-file", nargs="+")

        parsed_args, _ = parser.parse_known_args(list(cli_args))

        inputs = tuple(
            parsed_args.input
            if parsed_args.input_file is None
            else parsed_args.input_file
        )
        is_file = parsed_args.input_file is not None
        if is_file:
            for input_ in inputs:  # type: str
                with open(input_, "rb") as f:
                    if input_.endswith(".csv"):
                        content_type = "csv"
                    else:
                        content_type = "json"
                    yield InferenceTask(
                        context=InferenceContext(cli_args=cli_args),
                        data=(f.read(), content_type),
                    )

        else:
            for input_ in inputs:
                yield InferenceTask(
                    context=InferenceContext(cli_args=cli_args),
                    data=(input_, "csv" if parsed_args.csv else "json"),
                )

    def from_http_request(self, req: HTTPRequest) -> InferenceTask:
        content_type = (
            "csv" if req.parsed_headers.content_type == "text/csv" else "json"
        )

        return InferenceTask(
            data=(req.body, content_type),
            context=InferenceContext(http_headers=req.parsed_headers),
        )

    def from_aws_lambda_event(self, event: AwsLambdaEvent) -> InferenceTask:
        content_type = (
            "csv"
            if event.get("headers", {}).get("Content-Type", "application/json")
               == "text/csv"
            else "json"
        )

        return InferenceTask(
            context=InferenceContext(aws_lambda_event=event),
            data=(event.get("body"), content_type),
        )

    def extract_user_func_args(self, tasks: Iterable[InferenceTask]) -> ApiFuncArgs:
        dataframes = list()

        for task in tasks:
            task_bytes, content_type = task.data
            if content_type == "csv":
                df = pd.read_csv(task_bytes)
            else:
                try:
                    df = pd.read_json(task_bytes, orient=self.orient, typ=self.typ)
                except ValueError:
                    df = None
                    task.discard(
                        "Only the text/csv and application/json content types are "
                        "permitted at this endpoint",
                        http_status=400,
                    )
            if self.typ == "frame" and self.input_dtypes is not None:
                check_dataframe_column_contains(self.input_dtypes, df)

            dataframes.append(df)

        return dataframes

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
