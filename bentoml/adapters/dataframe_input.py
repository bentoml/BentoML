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
from bentoml.types import (
    AwsLambdaEvent,
    InferenceContext,
    InferenceTask,
    ParsedHeaders,
)
from bentoml.utils.dataframe_util import (
    PANDAS_DATAFRAME_TO_JSON_ORIENT_OPTIONS,
    read_dataframes_from_json_n_csv,
)
from bentoml.utils.lazy_loader import LazyLoader

pandas = LazyLoader('pandas', globals(), 'pandas')

DataFrameTask = InferenceTask[BinaryIO]
ApiFuncArgs = Tuple['pandas.DataFrame']


class DataframeInput(FileInput):
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
        if isinstance(self.input_dtypes, dict):
            json_schema = {  # For now, only declare JSON on docs.
                "schema": {
                    "type": "object",
                    "properties": {
                        k: {"type": "array", "items": {"type": self._get_type(v)}}
                        for k, v in self.input_dtypes.items()
                    },
                }
            }
        else:
            json_schema = {"schema": {"type": "object"}}
        return {
            "multipart/form-data": {
                "schema": {
                    "type": "object",
                    "properties": {"file": {"type": "string", "format": "binary"}},
                }
            },
            "application/json": json_schema,
            "text/csv": {"schema": {"type": "string", "format": "binary"}},
        }

    def from_aws_lambda_event(self, event: AwsLambdaEvent) -> InferenceTask[BinaryIO]:
        parsed_headers = ParsedHeaders.parse(tuple(event.get('headers', {}).items()))
        if parsed_headers.content_type == "text/csv":
            bio = io.BytesIO(event["body"].encode())
            bio.name = "input.csv"
        else:
            # Optimistically assuming Content-Type to be "application/json"
            bio = io.BytesIO(event["body"].encode())
            bio.name = "input.json"
        return InferenceTask(
            context=InferenceContext(aws_lambda_event=event), data=bio,
        )

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
        fmts, datas = tuple(
            zip(*((self._detect_format(task), task.data.read()) for task in tasks))
        )

        df, batchs = read_dataframes_from_json_n_csv(
            datas,
            fmts,
            orient=self.orient,
            columns=self.columns,
            dtype=self.input_dtypes,
        )

        if df is None:
            for task in tasks:
                task.discard(
                    http_status=400,
                    err_msg=f"{self.__class__.__name__} Wrong input format.",
                )
            return (df,)

        for task, batch, data in zip(tasks, batchs, datas):
            if batch == 0:
                task.discard(
                    http_status=400,
                    err_msg=f"{self.__class__.__name__} Wrong input format: {data}.",
                )
            else:
                task.batch = batch
        return (df,)
