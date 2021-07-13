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

from typing import Tuple, Type

from bentoml.types import (
    ApiFuncReturnValue,
    AwsLambdaEvent,
    HTTPResponse,
    InferenceResult,
    InferenceTask,
)

from .base_output import BaseOutputAdapter


def detect_suitable_adapter(result) -> Type[BaseOutputAdapter]:
    try:
        import pandas as pd

        if isinstance(result, (pd.DataFrame, pd.Series)):
            from .dataframe_output import DataframeOutput

            return DataframeOutput
    except ImportError:
        pass

    try:
        import tensorflow as tf

        if isinstance(result, tf.Tensor):
            from .tensorflow_tensor_output import TfTensorOutput

            return TfTensorOutput
    except ImportError:
        pass

    from .json_output import JsonOutput

    return JsonOutput


class DefaultOutput(BaseOutputAdapter):
    """
    Detect suitable output adapter automatically
    OutputAdapter converts returns of user defined API function into specific output,
    such as HTTP response, command line stdout or AWS Lambda event object.

    Args:
        cors (str): DEPRECATED. Moved to the configuration file.
            The value of the Access-Control-Allow-Origin header set in the
            HTTP/AWS Lambda response object. If set to None, the header will not be set.
            Default is None.
    """

    def __init__(self, **kwargs):
        super(DefaultOutput, self).__init__(**kwargs)
        self.actual_adapter = None
        from .json_output import JsonOutput

        self.backup_adapter = JsonOutput()

    def pack_user_func_return_value(
        self, return_result: ApiFuncReturnValue, tasks: Tuple[InferenceTask],
    ) -> Tuple[InferenceResult]:
        if self.actual_adapter is None:
            self.actual_adapter = detect_suitable_adapter(return_result)()
        if self.actual_adapter:
            return self.actual_adapter.pack_user_func_return_value(
                return_result, tasks=tasks
            )

        raise NotImplementedError()

    def to_http_response(self, result) -> HTTPResponse:
        if self.actual_adapter:
            return self.actual_adapter.to_http_response(result)
        return self.backup_adapter.to_http_response(result)

    def to_cli(self, results) -> int:
        if self.actual_adapter:
            return self.actual_adapter.to_cli(results)
        return self.backup_adapter.to_cli(results)

    def to_aws_lambda_event(self, result) -> AwsLambdaEvent:
        if self.actual_adapter:
            return self.actual_adapter.to_aws_lambda_event(result)
        return self.backup_adapter.to_aws_lambda_event(result)
