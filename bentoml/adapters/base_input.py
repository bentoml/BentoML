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

import json
from typing import Iterable

try:
    import pandas as pd
except ImportError:
    pd = None
import numpy as np

from bentoml import config as bentoml_config
from bentoml.marshal.utils import SimpleResponse, SimpleRequest

PANDAS_DATAFRAME_TO_DICT_ORIENT_OPTIONS = [
    'dict',
    'list',
    'series',
    'split',
    'records',
    'index',
]


class BaseInputAdapter:
    """InputAdapter is an abstraction layer between user defined API callback function
    and prediction request input in a variety of different forms, such as HTTP request
    body, command line arguments or AWS Lambda event object.
    """

    HTTP_METHODS = ["POST", "GET"]
    _BATCH_REQUEST_HEADER = bentoml_config("apiserver").get("batch_request_header")

    BATCH_MODE_SUPPORTED = False

    def __init__(self, output_adapter=None, **base_config):
        '''
        base_configs:
            - is_batch_input
        '''
        self._config = base_config
        self._output_adapter = output_adapter

    @property
    def config(self):
        if getattr(self, '_config', None) is None:
            self._config = {}
        return self._config

    @property
    def output_adapter(self):
        if self._output_adapter is None:
            from .default_output import DefaultOutput

            self._output_adapter = DefaultOutput()
        return self._output_adapter

    def handle_request(self, request, func):
        """Handles an HTTP request, convert it into corresponding data
        format that user API function is expecting, and return API
        function result as the HTTP response to client

        :param request: Flask request object
        :param func: user API function
        """
        raise NotImplementedError

    def handle_batch_request(
        self, requests: Iterable[SimpleRequest], func
    ) -> Iterable[SimpleResponse]:
        """Handles an HTTP request, convert it into corresponding data
        format that user API function is expecting, and return API
        function result as the HTTP response to client

        :param requests: List of flask request object
        :param func: user API function
        """
        raise NotImplementedError

    def handle_cli(self, args, func):
        """Handles an CLI command call, convert CLI arguments into
        corresponding data format that user API function is expecting, and
        prints the API function result to console output

        :param args: CLI arguments
        :param func: user API function
        """
        raise NotImplementedError

    def handle_aws_lambda_event(self, event, func):
        """Handles a Lambda event, convert event dict into corresponding
        data format that user API function is expecting, and use API
        function result as response

        :param event: AWS lambda event data of the python `dict` type
        :param func: user API function
        """
        raise NotImplementedError

    @property
    def request_schema(self):
        """
        :return: OpenAPI json schema for the HTTP API endpoint created with this input
                 adapter
        """
        return {"application/json": {"schema": {"type": "object"}}}

    @property
    def pip_dependencies(self):
        """
        :return: List of PyPI package names required by this InputAdapter
        """
        return []


class NumpyJsonEncoder(json.JSONEncoder):
    """ Special json encoder for numpy types """

    def default(self, o):  # pylint: disable=method-hidden
        if isinstance(o, np.generic):
            return o.item()

        if isinstance(o, np.ndarray):
            return o.tolist()

        return json.JSONEncoder.default(self, o)


def api_func_result_to_json(result, pandas_dataframe_orient="records"):
    assert (
        pandas_dataframe_orient in PANDAS_DATAFRAME_TO_DICT_ORIENT_OPTIONS
    ), f"unkown pandas dataframe orient '{pandas_dataframe_orient}'"

    if pd and isinstance(result, pd.DataFrame):
        return result.to_json(orient=pandas_dataframe_orient)

    if pd and isinstance(result, pd.Series):
        return pd.DataFrame(result).to_dict(orient=pandas_dataframe_orient)

    try:
        return json.dumps(result, cls=NumpyJsonEncoder)
    except (TypeError, OverflowError):
        # when result is not JSON serializable
        return json.dumps({"result": str(result)})
