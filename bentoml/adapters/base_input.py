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


from bentoml.marshal.utils import SimpleResponse, SimpleRequest, BATCH_REQUEST_HEADER


class BaseInputAdapter:
    """InputAdapter is an abstraction layer between user defined API callback function
    and prediction request input in a variety of different forms, such as HTTP request
    body, command line arguments or AWS Lambda event object.
    """

    HTTP_METHODS = ["POST", "GET"]

    BATCH_MODE_SUPPORTED = False

    def __init__(self, output_adapter=None, http_input_example=None, **base_config):
        '''
        base_configs:
            - is_batch_input
        '''
        self._config = base_config
        self._output_adapter = output_adapter
        self._http_input_example = http_input_example

    @property
    def config(self):
        if getattr(self, '_config', None) is None:
            self._config = {}
        return self._config

    def is_batch_request(self, request):
        if BATCH_REQUEST_HEADER in request.parsed_headers:
            return request.parsed_headers[BATCH_REQUEST_HEADER] != 'false'
        return self.config.get("is_batch_input", False)

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
