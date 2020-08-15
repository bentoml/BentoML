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

import os
import json
import argparse
import traceback
from typing import Iterable, List, Tuple
from json import JSONDecodeError

import flask

from bentoml.exceptions import BadInput
from bentoml.types import (
    HTTPRequest,
    HTTPResponse,
    JsonSerializable,
    AwsLambdaEvent,
    InferenceTask,
    InferenceContext,
    JSON_CHARSET,
)
from bentoml.adapters.base_input import BaseInputAdapter
from bentoml.adapters.utils import concat_list


UserArgs = Tuple[List[JsonSerializable]]


class JsonInput(BaseInputAdapter[UserArgs]):
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
        self, events: List[AwsLambdaEvent]
    ) -> Iterable[InferenceTask[bytes]]:
        return [
            InferenceTask(
                context=InferenceContext(aws_lambda_event=e),
                data=e['body'].encode(JSON_CHARSET),
            )
            for e in events
        ]

    def from_cli(self, cli_inputs, other_args) -> Iterable[InferenceTask[bytes]]:
        return [
            InferenceTask(context=InferenceContext(cli_args=other_args), data=i)
            for i in cli_inputs
        ]

    def extract_user_func_args(self, tasks: Iterable[InferenceTask[bytes]]) -> UserArgs:
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


class _JsonInput(BaseInputAdapter):
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

    def __init__(self, is_batch_input=False, **base_kwargs):
        super().__init__(is_batch_input=is_batch_input, **base_kwargs)

    def handle_request(self, request: flask.Request, func):
        if request.content_type != "application/json":
            raise BadInput(
                "Request content-type must be 'application/json' for this "
                "BentoService API"
            )
        resps = self.handle_batch_request(
            [HTTPRequest.from_flask_request(request)], func
        )
        return resps[0].to_flask_response()

    def handle_batch_request(
        self, requests: Iterable[HTTPRequest], func
    ) -> Iterable[HTTPResponse]:
        bad_resp = HTTPResponse(400, body="Bad Input")
        instances_list = [None] * len(requests)
        fallbacks = [bad_resp] * len(requests)
        batch_flags = [None] * len(requests)

        for i, request in enumerate(requests):
            batch_flags[i] = self.is_batch_request(request)
            try:
                raw_str = request.body
                parsed_json = json.loads(raw_str)
                instances_list[i] = parsed_json
            except (json.JSONDecodeError, UnicodeDecodeError):
                fallbacks[i] = HTTPResponse(400, body="Not a valid json")
            except Exception:  # pylint: disable=broad-except
                import traceback

                err = traceback.format_exc()
                fallbacks[i] = HTTPResponse(500, body=f"Internal Server Error: {err}")

        merged_instances, slices = concat_list(instances_list, batch_flags=batch_flags)
        merged_result = func(merged_instances)
        return self.output_adapter.to_batch_response(
            merged_result, slices, fallbacks, requests
        )

    def handle_cli(self, args, func):
        parser = argparse.ArgumentParser()
        parser.add_argument("--input", required=True)
        parsed_args, unknown_args = parser.parse_known_args(args)

        if os.path.isfile(parsed_args.input):
            with open(parsed_args.input, "r") as content_file:
                content = content_file.read()
        else:
            content = parsed_args.input

        input_json = json.loads(content)
        result = func([input_json])[0]
        return self.output_adapter.to_cli(result, unknown_args)

    def handle_aws_lambda_event(self, event, func):
        try:
            parsed_json = json.loads(event["body"])
        except JSONDecodeError:
            raise BadInput("Request body must contain valid json")

        result = func([parsed_json])[0]
        return self.output_adapter.to_aws_lambda_event(result, event)
