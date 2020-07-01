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
from typing import Iterable

import flask

from bentoml.exceptions import BadInput
from bentoml.marshal.utils import SimpleResponse, SimpleRequest
from bentoml.adapters.base_input import BaseInputAdapter
from bentoml.adapters.utils import concat_list


class JsonInput(BaseInputAdapter):
    """JsonInput parses REST API request or CLI command into parsed_json(a
    dict in python) and pass down to user defined API function

    """

    BATCH_MODE_SUPPORTED = True

    def __init__(self, is_batch_input=False, **base_kwargs):
        super(JsonInput, self).__init__(is_batch_input=is_batch_input, **base_kwargs)

    def handle_request(self, request: flask.Request, func):
        if request.content_type == "application/json":
            parsed_json = json.loads(request.get_data(as_text=True))
        else:
            raise BadInput(
                "Request content-type must be 'application/json' for this "
                "BentoService API"
            )

        result = func(parsed_json)
        return self.output_adapter.to_response(result, request)

    def handle_batch_request(
        self, requests: Iterable[SimpleRequest], func
    ) -> Iterable[SimpleResponse]:
        bad_resp = SimpleResponse(400, None, "Bad Input")
        instances_list = [None] * len(requests)
        fallbacks = [bad_resp] * len(requests)
        batch_flags = [None] * len(requests)

        for i, request in enumerate(requests):
            batch_flags[i] = self.is_batch_request(request)
            try:
                raw_str = request.data
                parsed_json = json.loads(raw_str)
                instances_list[i] = parsed_json
            except (json.JSONDecodeError, UnicodeDecodeError):
                fallbacks[i] = SimpleResponse(400, None, "Not a valid json")
            except Exception:  # pylint: disable=broad-except
                import traceback

                err = traceback.format_exc()
                fallbacks[i] = SimpleResponse(
                    500, None, f"Internal Server Error: {err}"
                )

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
        result = func(input_json)
        return self.output_adapter.to_cli(result, unknown_args)

    def handle_aws_lambda_event(self, event, func):
        if event["headers"]["Content-Type"] == "application/json":
            parsed_json = json.loads(event["body"])
        else:
            raise BadInput(
                "Request content-type must be 'application/json' for this "
                "BentoService API lambda endpoint"
            )

        result = func(parsed_json)
        return self.output_adapter.to_aws_lambda_event(result, event)
