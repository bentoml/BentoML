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

import json
import argparse
from typing import Iterable, List

from bentoml.types import (
    HTTPResponse,
    InferenceContext,
    DefaultErrorContext,
    InferenceResult,
    JsonSerializable,
)
from bentoml.adapters.utils import NumpyJsonEncoder
from bentoml.adapters.base_output import BaseOutputAdapter


UserReturnValue = List[JsonSerializable]


class JsonSerializableOutput(BaseOutputAdapter[UserReturnValue]):
    """
    Converts result of user defined API function into specific output.

    Args:
        cors (str): The value of the Access-Control-Allow-Origin header set in the
            AWS Lambda response object. Default is "*". If set to None,
            the header will not be set.
    """

    def pack_user_func_return_value(
        self, return_result: UserReturnValue, contexts: List[InferenceContext],
    ) -> List[InferenceResult[str]]:
        results = []
        for json_obj, context in zip(return_result, contexts):
            args = context.cli_args
            if args:
                parser = argparse.ArgumentParser()
                parser.add_argument(
                    "-o", "--output", default="str", choices=["str", "json"]
                )
                parsed_args, _ = parser.parse_known_args(args)
                output = parsed_args.output
            else:
                output = "json"
            try:
                if output == "json":
                    json_str = json.dumps(json_obj, cls=NumpyJsonEncoder)
                else:
                    json_str = str(json_obj)
                results.append(
                    InferenceResult(
                        data=json_str,
                        context=InferenceContext(
                            http_status=200,
                            http_headers=(("Content-Type", "application/json"),),
                        ),
                    )
                )
            except AssertionError as e:
                results.append(
                    InferenceResult(
                        context=DefaultErrorContext(err_msg=str(e), http_status=400,),
                    )
                )
            except Exception as e:  # pylint: disable=broad-except
                results.append(
                    InferenceResult(
                        context=DefaultErrorContext(err_msg=str(e), http_status=500,),
                    )
                )
        return results

    def to_http_response(self, results,) -> Iterable[HTTPResponse]:
        return (
            HTTPResponse(r.context.http_status, r.context.http_headers, r.data)
            for r in results
        )

    def to_cli(self, results):
        """Handles an CLI command call, convert CLI arguments into
        corresponding data format that user API function is expecting, and
        prints the API function result to console output

        :param args: CLI arguments
        """
        flag = 0
        for result in results:
            if result.context.err_msg:
                print(result.context.err_msg)
                flag = 1
            else:
                print(result.data)
        return flag

    def to_aws_lambda_event(self, results):

        # Allow disabling CORS by setting it to None
        for result in results:
            if self.cors:
                yield {
                    "statusCode": result.context.http_status,
                    "body": result.data,
                    "headers": {"Access-Control-Allow-Origin": self.cors},
                }

            yield {"statusCode": result.context.http_status, "body": result.data}
