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

from typing import Iterable, List, Generic

from bentoml.types import (
    AwsLambdaEvent,
    HTTPResponse,
    ApiFuncReturnValue,
    InferenceResult,
    InferenceContext,
)


class BaseOutputAdapter(Generic[ApiFuncReturnValue]):
    """
    OutputAdapter is an layer between result of user defined API callback function
    and final output in a variety of different forms,
    such as HTTP response, command line stdout or AWS Lambda event object.
    """

    def __init__(self, cors='*'):
        self.cors = cors

    @property
    def config(self):
        return dict(cors=self.cors,)

    @property
    def pip_dependencies(self):
        """
        :return: List of PyPI package names required by this OutputAdapter
        """
        return []

    def pack_user_func_return_value(
        self, return_result: ApiFuncReturnValue, contexts: List[InferenceContext],
    ) -> List[InferenceResult]:
        """
        Pack the return value of user defined API function into InferenceResults
        """
        raise NotImplementedError()

    def to_http_response(
        self, results: Iterable[InferenceResult]
    ) -> Iterable[HTTPResponse]:
        """
        Converts InferenceResults into HTTP responses.
        """
        raise NotImplementedError()

    def to_cli(self, results: Iterable[InferenceResult]) -> int:
        """
        Converts InferenceResults into CLI output.
        """
        raise NotImplementedError()

    def to_aws_lambda_event(
        self, results: Iterable[InferenceResult]
    ) -> Iterable[AwsLambdaEvent]:
        """
        Converts InferenceResults into AWS lambda events.
        """
        raise NotImplementedError()
