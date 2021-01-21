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

import base64
from typing import Iterable, Sequence

from bentoml.adapters.base_output import BaseOutputAdapter, regroup_return_value
from bentoml.types import AwsLambdaEvent, HTTPResponse, InferenceResult, InferenceTask

ApiFuncReturnValue = Sequence[bytes]


class FileOutput(BaseOutputAdapter):
    """
    Converts result of user defined API function into specific output.

    Args:
        cors (str): The value of the Access-Control-Allow-Origin header set in the
            AWS Lambda response object. Default is "*". If set to None,
            the header will not be set.
    """

    def pack_user_func_return_value(
        self, return_result: ApiFuncReturnValue, tasks: Sequence[InferenceTask],
    ) -> Sequence[InferenceResult[str]]:
        results = []
        for bytes_, task in regroup_return_value(return_result, tasks):
            try:
                assert isinstance(bytes_, bytes), "returned value should be bytes"
                results.append(
                    InferenceResult(
                        data=bytes_,
                        http_status=200,
                        http_headers={"Content-Type": "application/octet-stream"},
                    )
                )
            except AssertionError as e:
                task.discard(err_msg=str(e), http_status=400)
            except Exception as e:  # pylint: disable=broad-except
                task.discard(err_msg=str(e), http_status=500)
        return tuple(results)

    def to_http_response(self, result: InferenceResult) -> HTTPResponse:
        return HTTPResponse(
            status=result.http_status,
            headers=tuple(result.http_headers.items()),
            body=result.err_msg or result.data,
        )

    def to_cli(self, results: Iterable[InferenceResult]) -> int:
        """Handles an CLI command call, convert CLI arguments into
        corresponding data format that user API function is expecting, and
        prints the API function result to console output

        :param args: CLI arguments
        """
        flag = 0
        for result in results:
            if result.err_msg:
                print(result.err_msg)
                flag = 1
            else:
                print(result.data)
        return flag

    def to_aws_lambda_event(self, result: InferenceResult) -> AwsLambdaEvent:
        if not result.err_msg:
            b64_data = {"b64": base64.b64encode(result.data)}
        else:
            b64_data = None
        # Allow disabling CORS by setting it to None
        if self.cors:
            return {
                "statusCode": result.http_status,
                "body": result.err_msg or b64_data,
                "headers": {"Access-Control-Allow-Origin": self.cors},
            }
        else:
            return {
                "statusCode": result.http_status,
                "body": result.err_msg or b64_data,
            }
