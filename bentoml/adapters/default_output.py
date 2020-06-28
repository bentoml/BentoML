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

from bentoml.marshal.utils import SimpleResponse, SimpleRequest
from .base_output import BaseOutputAdapter


def detect_suitable_adapter(result, slices=None):
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

    if slices is not None:
        for s in slices:
            if s:
                return detect_suitable_adapter(result[s])
    if isinstance(result, (list, tuple)):
        return detect_suitable_adapter(result[0])

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

    def to_batch_response(
        self,
        result_conc,
        slices=None,
        fallbacks=None,
        requests: Iterable[SimpleRequest] = None,
    ) -> Iterable[SimpleResponse]:
        """Converts corresponding data merged by batching service into HTTP responses

        :param result_conc: result of user API function
        :param slices: auto-batching slices
        :param requests: request objects
        """
        if self.actual_adapter is None:
            self.actual_adapter = detect_suitable_adapter(result_conc, slices)()
        return self.actual_adapter.to_batch_response(
            result_conc, slices, fallbacks, requests
        )

    def to_cli(self, result, args):
        """Converts corresponding data into an CLI output.

        :param result: result of user API function
        :param args: CLI args
        """
        if self.actual_adapter is None:
            self.actual_adapter = detect_suitable_adapter(result)()
        return self.actual_adapter.to_cli(result, args)

    def to_aws_lambda_event(self, result, event):
        """Converts corresponding data into a Lambda event.

        :param result: result of user API function
        :param event: input event
        """
        if self.actual_adapter is None:
            self.actual_adapter = detect_suitable_adapter(result)()
        return self.actual_adapter.to_aws_lambda_event(result, event)
