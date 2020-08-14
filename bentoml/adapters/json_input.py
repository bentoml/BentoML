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
import inspect
import functools
import argparse
import traceback
from typing import Iterable, List, Union, Tuple, overload
from json import JSONDecodeError

import uuid
import flask

from bentoml.exceptions import BadInput
from bentoml.types import (
    Input,
    UserArg,
    UserReturnValue,
    HTTPRequest,
    HTTPResponse,
    JsonSerializable,
    AwsLambdaEvent,
    InferenceTask,
    InferenceResult,
    InferenceContext,
)
from bentoml.adapters.base_input import BaseInputAdapter
from bentoml.adapters.base_output import BaseOutputAdapter
from bentoml.adapters.utils import concat_list


class InferenceAPI:
    def __init__(
        self,
        input_adapter: BaseInputAdapter,
        output_adapter: BaseOutputAdapter,
        user_func: callable,
    ):
        self.input_adapter = input_adapter
        self.output_adapter = output_adapter

        # allow user to define handlers without 'contexts' kwargs
        _sig = inspect.signature(user_func)
        try:
            _sig.bind(contexts=None)
            self.user_func = user_func
        except TypeError:

            @functools.wraps(user_func)
            def safe_user_func(*args, **kwargs):
                kwargs.pop('contexts')
                return user_func(*args, **kwargs)

            self.user_func = safe_user_func

    def infer(self, inf_tasks: Iterable[InferenceTask]) -> List[InferenceResult]:
        # task validation
        def _validation(inf_tasks):
            for task in inf_tasks:
                if self.input_adapter.validate_task(task):
                    yield task
                else:
                    task.discard(http_status=400, err_msg="Input validation failed")

        # extract args
        user_args = self.input_adapter.extract_user_func_args(_validation(inf_tasks))
        contexts = [t.contexts for t in inf_tasks if not t.is_discarded]

        # call user function
        user_return = self.user_func(*user_args, contexts=contexts)

        # pack return value
        inf_results = self.output_adapter.pack_user_func_return_value(
            user_return, contexts=contexts
        )
        full_results = InferenceResult.complete_discarded(inf_tasks, inf_results)
        return full_results

    def handle_http_request(
        self, reqs: Iterable[HTTPRequest]
    ) -> Iterable[HTTPResponse]:
        inf_tasks = self.input_adapter.from_http_request(reqs)
        results = self.infer(inf_tasks)
        return self.output_adapter.to_batch_response(results)

    def handle_aws_lambda_event(
        self, events: List[AwsLambdaEvent]
    ) -> Iterable[AwsLambdaEvent]:
        inf_tasks = self.input_adapter.from_aws_lambda_event(events)
        results = self.infer(inf_tasks)
        return self.output_adapter.to_aws_lambda_event(results)

    def handle_cli(self, args: List[str]) -> int:
        parser = argparse.ArgumentParser()
        input_g = parser.add_mutually_exclusive_group(required=True)
        input_g.add_argument('--input', nargs="+", type=bytes)
        input_g.add_argument('--input-file', nargs="+")

        parser.add_argument("--max-batch-size", default=None, type=int)
        parsed_args, other_args = parser.parse_known_args(args)

        input_args = (
            parsed_args.input
            if parsed_args.input_file is None
            else parsed_args.input_file
        )
        is_file = parsed_args.input_file is not None

        batch_size = parsed_args.batch_size or len(input_args)

        for i in range(0, len(input_args), batch_size):
            cli_inputs = input_args[i : i + batch_size]
            if is_file:
                cli_inputs = [open(file_path, 'rb').read() for file_path in cli_inputs]
            tasks = self.input_adapter.from_cli(cli_inputs, other_args)
            results = self.infer(tasks)
            self.output_adapter.to_cli(results)

        return 0


class BaseInputAdapter:
    BATCH_MODE_SUPPORTED = True

    def from_http_request(
        self, reqs: Iterable[HTTPRequest]
    ) -> Iterable[InferenceTask[str]]:
        raise NotImplementedError()

    def from_aws_lambda_event(
        self, events: List[AwsLambdaEvent]
    ) -> Iterable[InferenceTask[str]]:
        raise NotImplementedError()

    def from_cli(self, cli_inputs, other_args) -> Iterable[InferenceTask[str]]:
        raise NotImplementedError()

    def validate_task(self, _: InferenceTask):
        raise NotImplementedError()

    def extract_user_func_args(
        self, tasks: Iterable[InferenceTask]
    ) -> Tuple[UserArg, ...]:
        raise NotImplementedError()


class BaseOutputAdapter:
    @overload
    def pack_user_func_return_value(
        self, return_value: UserReturnValue, contexts: List[InferenceContext]
    ) -> List[InferenceResult]:
        pass

    @overload
    def pack_user_func_return_value(
        self, return_result: List[InferenceResult], contexts: List[InferenceContext]
    ) -> List[InferenceResult]:
        pass

    def pack_user_func_return_value(
        self, return_result, contexts
    ) -> List[InferenceResult]:
        raise NotImplementedError()

    def to_http_request(self, results: Iterable[InferenceResult]):
        raise NotImplementedError()

    def to_cli(self, results: Iterable[InferenceResult]):
        raise NotImplementedError()

    def to_aws_lambda_event(self, results: Iterable[InferenceResult]):
        raise NotImplementedError()


class JsonInput(BaseInputAdapter):
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
    ) -> List[InferenceTask[str]]:
        return [
            InferenceTask(
                context=dict(inf_id=uuid.uuid4(), http_headers=r.parsed_headers),
                data=r.body,
            )
            for r in reqs
        ]

    def from_aws_lambda_event(
        self, events: List[AwsLambdaEvent]
    ) -> Iterable[InferenceTask[str]]:
        return [
            InferenceTask(
                context=dict(inf_id=uuid.uuid4(), aws_event=e), data=e['body'],
            )
            for e in events
        ]

    def from_cli(self, cli_inputs, other_args) -> Iterable[InferenceTask[str]]:
        parser = argparse.ArgumentParser()
        parser.add_argument("--charset", default="utf-8", type=str)
        parsed_args, _ = parser.parse_known_args(other_args)
        return [
            InferenceTask(
                context=dict(inf_id=uuid.uuid4(), cli_args=other_args),
                data=i.decode(parsed_args.charset),
            )
            for i in cli_inputs
        ]

    def extract_user_func_args(
        self, tasks: Iterable[InferenceTask[str]]
    ) -> Iterable[JsonSerializable]:
        json_inputs = []
        for task in tasks:
            try:
                parsed_json = json.loads(task.data)
                json_inputs.append(parsed_json)
            except (json.JSONDecodeError, UnicodeDecodeError):
                task.discard(http_status=400, err_msg="Not a valid JSON input")
            except Exception:  # pylint: disable=broad-except
                err = traceback.format_exc()
                task.discard(http_status=500, err_msg=f"Internal Server Error: {err}")
        return json_inputs


class JsonInput(BaseInputAdapter):
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
        super(JsonInput, self).__init__(is_batch_input=is_batch_input, **base_kwargs)

    def handle_request(self, request: flask.Request):
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
