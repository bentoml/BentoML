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
from typing import Iterable, Iterator, Mapping, Optional, Sequence, Tuple

from bentoml.adapters.string_input import StringInput
from bentoml.exceptions import MissingDependencyException
from bentoml.types import HTTPHeaders, InferenceTask
from bentoml.utils.dataframe_util import (
    PANDAS_DATAFRAME_TO_JSON_ORIENT_OPTIONS,
    read_dataframes_from_json_n_csv,
)
from bentoml.utils.lazy_loader import LazyLoader

pandas = LazyLoader('pandas', globals(), 'pandas')

DataFrameTask = InferenceTask[str]
ApiFuncArgs = Tuple['pandas.DataFrame']


class DataframeInput(StringInput):
    """
    Convert various inputs(HTTP, Aws Lambda or CLI) to pandas dataframe, passing it to
    API functions.

    Parameters
    ----------
    orient : str
        Indication of expected JSON string format.
        Compatible JSON strings can be produced by ``to_json()`` with a
        corresponding orient value.
        The set of possible orients is:

        - ``'split'`` : dict like
          ``{index -> [index], columns -> [columns], data -> [values]}``
        - ``'records'`` : list like
          ``[{column -> value}, ... , {column -> value}]``
        - ``'index'`` : dict like ``{index -> {column -> value}}``
        - ``'columns'`` : dict like ``{column -> {index -> value}}``
        - ``'values'`` : just the values array

        The allowed and default values depend on the value
        of the `typ` parameter.

        * when ``typ == 'series'`` (not available now),

          - allowed orients are ``{'split','records','index'}``
          - default is ``'index'``
          - The Series index must be unique for orient ``'index'``.

        * when ``typ == 'frame'``,

          - allowed orients are ``{'split','records','index',
            'columns','values'}``
          - default is ``'columns'``
          - The DataFrame index must be unique for orients ``'index'`` and
            ``'columns'``.
          - The DataFrame columns must be unique for orients ``'index'``,
            ``'columns'``, and ``'records'``.

    typ : {'frame', 'series'}, default 'frame'
        ** Note: 'series' is not supported now. **
        The type of object to recover.

    dtype : dict, default None
        If is None, infer dtypes; if a dict of column to dtype, then use those.
        Not applicable for ``orient='table'``.

    input_dtypes : dict, default None
        ** Deprecated **
        The same as the `dtype`

    Raises
    -------
    ValueError: Incoming data is missing required columns in dtype

    ValueError: Incoming data format can not be handled. Only json and csv

    Examples
    -------
    Example Service:

    .. code-block:: python

        from bentoml import env, artifacts, api, BentoService
        from bentoml.adapters import DataframeInput
        from bentoml.frameworks.sklearn import SklearnModelArtifact

        @env(infer_pip_packages=True)
        @artifacts([SklearnModelArtifact('model')])
        class IrisClassifier(BentoService):

            @api(
                input=DataframeInput(
                    orient="records",
                    columns=["sw", "sl", "pw", "pl"],
                    dtype={"sw": "float", "sl": "float", "pw": "float", "pl": "float"},
                ),
                batch=True,
            )
            def predict(self, df):
                # Optional pre-processing, post-processing code goes here
                return self.artifacts.model.predict(df)

    Query with HTTP request::

        curl -i \\
          --header "Content-Type: application/json" \\
          --request POST \\
          --data '[{"sw": 1, "sl": 2, "pw": 1, "pl": 2}]' \\
          localhost:5000/predict

    OR::

        curl -i \\
          --header "Content-Type: text/csv" \\
          --request POST \\
          --data-binary @file.csv \\
          localhost:5000/predict

    Query with CLI command::

        bentoml run IrisClassifier:latest predict --input \\
          '[{"sw": 1, "sl": 2, "pw": 1, "pl": 2}]'

    OR::

        bentoml run IrisClassifier:latest predict --format csv --input-file test.csv

    """

    SINGLE_MODE_SUPPORTED = False

    def __init__(
        self,
        typ: str = "frame",
        orient: Optional[str] = None,
        columns: Sequence[str] = None,
        dtype: Mapping[str, object] = None,
        input_dtypes: Mapping[str, object] = None,
        **base_kwargs,
    ):
        super().__init__(**base_kwargs)
        dtype = dtype if dtype is not None else input_dtypes

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
            columns is None or dtype is None or set(dtype) == set(columns)
        ), "dtype must match columns"

        self.typ = typ
        self.orient = orient
        self.columns = columns
        if isinstance(dtype, (list, tuple)):
            self.dtype = dict((index, dtype) for index, dtype in enumerate(dtype))
        else:
            self.dtype = dtype

    @property
    def pip_dependencies(self):
        return ['pandas']

    @property
    def config(self):
        base_config = super().config
        return dict(base_config, orient=self.orient, typ=self.typ, dtype=self.dtype,)

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
        if isinstance(self.dtype, dict):
            json_schema = {  # For now, only declare JSON on docs.
                "schema": {
                    "type": "object",
                    "properties": {
                        k: {"type": "array", "items": {"type": self._get_type(v)}}
                        for k, v in self.dtype.items()
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

    @classmethod
    def _detect_format(cls, task: InferenceTask) -> str:
        if task.aws_lambda_event:
            headers = HTTPHeaders.from_dict(task.aws_lambda_event.get('headers', {}))
            if headers.content_type == "application/json":
                return "json"
            if headers.content_type == "text/csv":
                return "csv"
        elif task.http_headers:
            headers = task.http_headers
            if headers.content_type == "application/json":
                return "json"
            if headers.content_type == "text/csv":
                return "csv"
        elif task.cli_args:
            parser = argparse.ArgumentParser()
            parser.add_argument('--format', type=str, choices=['csv', 'json'])
            parsed_args, _ = parser.parse_known_args(list(task.cli_args))
            return parsed_args.format or "json"

        return "json"

    def extract_user_func_args(
        self, tasks: Iterable[InferenceTask[str]]
    ) -> ApiFuncArgs:
        fmts, datas = tuple(
            zip(*((self._detect_format(task), task.data) for task in tasks))
        )

        df, batchs = read_dataframes_from_json_n_csv(
            datas, fmts, orient=self.orient, columns=self.columns, dtype=self.dtype,
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

    def from_inference_job(
        self, input_=None, input_file=None, **extra_args,
    ) -> Iterator[InferenceTask[str]]:
        # TODO: generate small batches of InferenceTasks from large input files
        return super().from_inference_job(
            input_=input_, input_file=input_file, **extra_args
        )
