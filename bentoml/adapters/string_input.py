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

from typing import Iterable, Iterator, Sequence, Tuple

from bentoml.adapters.base_input import BaseInputAdapter, parse_cli_input
from bentoml.adapters.utils import decompress_gzip_request
from bentoml.types import AwsLambdaEvent, HTTPRequest, InferenceTask

ApiFuncArgs = Tuple[
    Sequence[str],
]


class StringInput(BaseInputAdapter):
    """
    Convert various inputs(HTTP, Aws Lambda or CLI) to strings(list of str), passing it
    to API functions.

    Parameters
    ----------
    none

    Errors
    -------
        400 UnicodeDecodeError
        400 Unsupported charset

    Example Request
    -------
    For clients, the request is the same as LegacyJsonInput, each includes single json.
        ```
        curl -i \
            --header "Content-Type: text/plain; charset=utf-8" \
            --request POST \
            --data 'best movie ever' \
            localhost:5000/predict
        ```
    """

    BATCH_MODE_SUPPORTED = True

    @decompress_gzip_request
    def from_http_request(self, req: HTTPRequest) -> InferenceTask[str]:
        try:
            return InferenceTask(
                http_headers=req.headers,
                data=req.body.decode(req.headers.charset or "utf-8"),
            )
        except UnicodeDecodeError:
            return InferenceTask().discard(
                http_status=400,
                err_msg=f"{self.__class__.__name__}: UnicodeDecodeError for {req.body}",
            )
        except LookupError:
            return InferenceTask().discard(
                http_status=400,
                err_msg=f"{self.__class__.__name__}: Unsupported charset {req.charset}",
            )

    def from_aws_lambda_event(self, event: AwsLambdaEvent) -> InferenceTask[str]:
        return InferenceTask(aws_lambda_event=event, data=event.get('body', ""),)

    def from_cli(self, cli_args: Tuple[str]) -> Iterator[InferenceTask[str]]:
        for input_ in parse_cli_input(cli_args):
            try:
                yield InferenceTask(
                    cli_args=cli_args,
                    data=input_.read().decode(),  # charset for cli input
                )
            except UnicodeDecodeError:
                yield InferenceTask().discard(
                    http_status=400,
                    err_msg=f"{self.__class__.__name__}: UnicodeDecodeError",
                )

    def extract_user_func_args(
        self, tasks: Iterable[InferenceTask[str]]
    ) -> ApiFuncArgs:
        return [task.data for task in tasks]
