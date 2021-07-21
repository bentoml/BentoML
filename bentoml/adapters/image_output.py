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

from typing import Iterable, Sequence

from bentoml.adapters.base_output import (BaseOutputAdapter,
                                          regroup_return_value)
from bentoml.adapters.utils import get_default_accept_image_formats
from bentoml.types import (AwsLambdaEvent, HTTPResponse, InferenceError,
                           InferenceResult, InferenceTask)
from bentoml.utils.lazy_loader import LazyLoader

# BentoML optional dependencies, using lazy load to avoid ImportError
Image = LazyLoader('Image', globals(), 'PIL.Image')
io = LazyLoader('io', globals(), 'io')

ApiFuncReturnValue = Sequence[bytes]


class ImageOutput(BaseOutputAdapter):
    """
    Converts result of user defined API function into image output.

        Args:
                cors (str): The value of the Access-Control-Allow-Origin header set in the AWS Lambda response object.
                        Default is "*". If set to None, the header will not be set.
                extension_format (str): Refers to the "Content-Type" value of the returned image. Default is "None".
                        If set to None, an attempt is made to retrieve the "Content-Type" value from the incoming data.
    """

    def __init__(self, extension_format: str = None, **kwargs):
        super().__init__(**kwargs)
        self.extension_format = extension_format

    def pack_user_func_return_value(
            self, return_result: ApiFuncReturnValue, tasks: Sequence[InferenceTask],
    ) -> Sequence[InferenceResult[str]]:
        """
        Pack the return value of user defined API function into InferenceResults
        """
        results = []
        for arrays_, task in regroup_return_value(return_result, tasks):
            try:
                if self.extension_format is None and task.http_headers.get('Content-Type', None).lower() in get_default_accept_image_formats():
                    self.extension_format = task.http_headers.get(
                        'Content-Type', None).lower()

                elif self.extension_format is not None and self.extension_format.lower() in get_default_accept_image_formats():
                    self.extension_format = self.extension_format.lower()

                else:
                    results.append(InferenceError(
                        err_msg=f"Current service only returns "
                        f"{get_default_accept_image_formats()} formats", http_status=400,))

                out_img = Image.fromarray(arrays_)
                buf = io.BytesIO()
                out_img.save(buf, format=self.extension_format[1:])

                results.append(
                    InferenceResult(
                        data=buf.getvalue(),
                        http_status=200,
                        http_headers={
                            "Content-Type": f"image/{self.extension_format}"},
                    )
                )

            except AssertionError as e:
                results.append(InferenceError(
                    err_msg=str(e), http_status=400,))
            except Exception as e:  # pylint: disable=broad-except
                results.append(InferenceError(
                    err_msg=str(e), http_status=500,))
        return tuple(results)

    def to_http_response(self, result: InferenceResult) -> HTTPResponse:
        """
        Converts InferenceResults into HTTP responses.
        """
        return HTTPResponse(
            status=result.http_status,
            headers=tuple(result.http_headers.items()),
            body=result.err_msg or result.data,
        )

    def to_cli(self, results: Iterable[InferenceResult]) -> int:
        """
        Converts InferenceResults into CLI output.
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
        """
        Converts InferenceResults into AWS lambda events.
        """
        # Allow disabling CORS by setting it to None
        if self.cors:
            return {
                "statusCode": result.http_status,
                "body": result.err_msg or result.data,
                "headers": {"Access-Control-Allow-Origin": self.cors},
            }
        else:
            return {
                "statusCode": result.http_status,
                "body": result.err_msg or result.data,
            }
