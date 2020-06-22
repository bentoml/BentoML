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

import flask

from bentoml.marshal.utils import SimpleResponse, SimpleRequest


class BaseOutputAdapter:
    """OutputAdapter is an layer between result of user defined API callback function
    and final output in a variety of different forms,
    such as HTTP response, command line stdout or AWS Lambda event object.
    """

    def __init__(self, cors='*'):
        self.cors = cors

    @property
    def config(self):
        return dict(cors=self.cors,)

    def to_response(self, result, request: flask.Request) -> flask.Response:
        """Converts corresponding data into an HTTP response

        :param result: result of user API function
        :param request: request object
        """
        simple_req = SimpleRequest.from_flask_request(request)
        simple_resp = self.to_batch_response((result,), requests=(simple_req,))[0]
        return simple_resp.to_flask_response()

    def to_batch_response(
        self, result_conc, slices=None, fallbacks=None, requests=None,
    ) -> Iterable[SimpleResponse]:
        """Converts corresponding data merged by batching service into HTTP responses

        :param result_conc: result of user API function
        :param slices: auto-batching slices
        :param requests: request objects
        """
        raise NotImplementedError()

    def to_cli(self, result, args):
        """Converts corresponding data into an CLI output.

        :param result: result of user API function
        :param args: CLI args
        """
        raise NotImplementedError()

    def to_aws_lambda_event(self, result, event):
        """Converts corresponding data into a Lambda event.

        :param result: result of user API function
        :param event: input event
        """
        raise NotImplementedError

    @property
    def pip_dependencies(self):
        """
        :return: List of PyPI package names required by this OutputAdapter
        """
        return []
