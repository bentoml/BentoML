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

from typing import Iterable, Generic, List

from bentoml.types import (
    UserArgs,
    HTTPRequest,
    InferenceTask,
    AwsLambdaEvent,
)


class BaseInputAdapter(Generic[UserArgs]):
    """
    InputAdapter is an abstraction layer between user defined API callback function
    and prediction request input in a variety of different forms, such as HTTP request
    body, command line arguments or AWS Lambda event object.
    """

    HTTP_METHODS = ["POST", "GET"]
    BATCH_MODE_SUPPORTED = False

    def __init__(self, http_input_example=None, **base_config):
        self._config = base_config
        self._http_input_example = http_input_example

    @property
    def config(self):
        return self._config

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

    def from_http_request(self, reqs: Iterable[HTTPRequest]) -> Iterable[InferenceTask]:
        """
        Handles HTTP requests, convert it into InferenceTasks
        """
        raise NotImplementedError()

    def from_aws_lambda_event(
        self, events: List[AwsLambdaEvent]
    ) -> Iterable[InferenceTask]:
        """
        Handles AWS lambda events, convert it into InferenceTasks
        """
        raise NotImplementedError()

    def from_cli(
        self, cli_inputs: Iterable[bytes], other_args: List[str]
    ) -> Iterable[InferenceTask]:
        """
        Handles CLI command, generate InferenceTasks
        """
        raise NotImplementedError()

    def validate_task(self, _: InferenceTask):
        return True

    def extract_user_func_args(self, tasks: Iterable[InferenceTask]) -> UserArgs:
        """
        Extract args that user API function is expecting from InferenceTasks
        """
        raise NotImplementedError()
