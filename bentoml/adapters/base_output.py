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

import itertools
from typing import Iterable, Iterator, Sequence, Tuple

from bentoml.types import (
    ApiFuncReturnValue,
    AwsLambdaEvent,
    HTTPResponse,
    InferenceResult,
    InferenceTask,
    Output,
)


def regroup_return_value(
    return_values: Iterable, tasks: Sequence[InferenceTask]
) -> Iterator[Tuple[Output, InferenceTask]]:
    iter_rv = iter(return_values)
    try:
        for task in tasks:
            if task.batch is None:
                yield next(iter_rv), task
            else:
                yield tuple(itertools.islice(iter_rv, task.batch)), task
    except StopIteration:
        for task in tasks:
            task.discard(
                http_status=500,
                err_msg="The return values of Api Function doesn't match tasks",
            )


class BaseOutputAdapter:
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
        self, return_result: ApiFuncReturnValue, tasks: Sequence[InferenceTask],
    ) -> Sequence[InferenceResult]:
        """
        Pack the return value of user defined API function into InferenceResults
        """
        raise NotImplementedError()

    def to_http_response(self, result: InferenceResult) -> HTTPResponse:
        """
        Converts InferenceResults into HTTP responses.
        """
        raise NotImplementedError()

    def to_cli(self, results: Iterable[InferenceResult]) -> int:
        """
        Converts InferenceResults into CLI output.
        """
        raise NotImplementedError()

    def to_aws_lambda_event(self, result: InferenceResult) -> AwsLambdaEvent:
        """
        Converts InferenceResults into AWS lambda events.
        """
        raise NotImplementedError()
