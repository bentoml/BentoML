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

from typing import Iterable, List

from bentoml.types import (
    UserReturnValue,
    HTTPResponse,
    InferenceResult,
    InferenceContext,
)
from .base_output import BaseOutputAdapter


def detect_suitable_adapter(result):
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

    # if isinstance(result, (list, tuple)):
    # return detect_suitable_adapter(result[0])

    from .json_output import JsonSerializableOutput

    return JsonSerializableOutput


class DefaultOutput(BaseOutputAdapter):
    """
    Detect suitable output adapter automatically and
    converts result of use defined API function into specific output.

    Args:
        cors (str): The value of the Access-Control-Allow-Origin header set in the
            AWS Lambda response object. Default is "*". If set to None,
            the header will not be set.
    """

    def __init__(self, **kwargs):
        super(DefaultOutput, self).__init__(**kwargs)
        self.actual_adapter = None

    def pack_user_func_return_value(
        self, return_result: UserReturnValue, contexts: List[InferenceContext],
    ) -> List[InferenceResult]:
        if self.actual_adapter is None:
            self.actual_adapter = detect_suitable_adapter(return_result)()
        if self.actual_adapter:
            return self.actual_adapter.pack_user_func_return_value(
                return_result, contexts=contexts
            )

        raise NotImplementedError()

    def to_http_response(self, results) -> Iterable[HTTPResponse]:
        """Converts corresponding data merged by batching service into HTTP responses

        :param result_conc: result of user API function
        :param slices: auto-batching slices
        :param requests: request objects
        """
        return self.actual_adapter.to_http_response(results)

    def to_cli(self, results):
        """Converts corresponding data into an CLI output.

        :param result: result of user API function
        :param args: CLI args
        """
        return self.actual_adapter.to_cli(results)

    def to_aws_lambda_event(self, results):
        """Converts corresponding data into a Lambda event.

        :param result: result of user API function
        :param event: input event
        """
        return self.actual_adapter.to_aws_lambda_event(results)
