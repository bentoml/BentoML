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
import traceback
from typing import Iterable, List, Tuple, Iterator


from bentoml.types import (
    HTTPRequest,
    JsonSerializable,
    AwsLambdaEvent,
    InferenceTask,
    InferenceContext,
    JSON_CHARSET,
)
from bentoml.adapters.base_input import BaseInputAdapter, parse_cli_input


ApiFuncArgs = Tuple[List[JsonSerializable]]


class JsonInput(BaseInputAdapter[ApiFuncArgs]):
    """JsonInput parses REST API request or CLI command into parsed_jsons(a list of
    json serializable object in python) and pass down to user defined API function

    ****
    How to upgrade from LegacyJsonInput(JsonInput before 0.8.3)

    To enable micro batching for API with json inputs, custom bento service should use
    JsonInput and modify the handler method like this:
        ```
        @bentoml.api(input=LegacyJsonInput())
        def predict(self, parsed_json):
            results = self.artifacts.classifier([parsed_json['text']])
            return results[0]
        ```
    --->
        ```
        @bentoml.api(input=JsonInput())
        def predict(self, parsed_jsons):
            results = self.artifacts.classifier([j['text'] for j in parsed_jsons])
            return results
        ```
    For clients, the request is the same as LegacyJsonInput, each includes single json.
        ```
        curl -i \
            --header "Content-Type: application/json" \
            --request POST \
            --data '{"text": "best movie ever"}' \
            localhost:5000/predict
        ```
    """

    BATCH_MODE_SUPPORTED = True

    def from_http_request(
        self, reqs: Iterable[HTTPRequest]
    ) -> List[InferenceTask[bytes]]:
        return [
            InferenceTask(
                context=InferenceContext(http_headers=r.parsed_headers), data=r.body,
            )
            for r in reqs
        ]

    def from_aws_lambda_event(
        self, events: Iterable[AwsLambdaEvent]
    ) -> List[InferenceTask[bytes]]:
        return [
            InferenceTask(
                context=InferenceContext(aws_lambda_event=e),
                data=e['body'].encode(JSON_CHARSET),
            )
            for e in events
        ]

    def from_cli(self, cli_args: List[str]) -> Iterator[InferenceTask[bytes]]:
        for i in parse_cli_input(cli_args):
            yield InferenceTask(context=InferenceContext(cli_args=cli_args), data=i)

    def extract_user_func_args(
        self, tasks: Iterable[InferenceTask[bytes]]
    ) -> ApiFuncArgs:
        json_inputs = []
        for task in tasks:
            try:
                json_str = task.data.decode(JSON_CHARSET)
                parsed_json = json.loads(json_str)
                json_inputs.append(parsed_json)
            except UnicodeDecodeError:
                task.discard(
                    http_status=400, err_msg=f"JSON must be encoded in {JSON_CHARSET}"
                )
            except json.JSONDecodeError:
                task.discard(http_status=400, err_msg="Not a valid JSON format")
            except Exception:  # pylint: disable=broad-except
                err = traceback.format_exc()
                task.discard(http_status=500, err_msg=f"Internal Server Error: {err}")
        return (json_inputs,)
