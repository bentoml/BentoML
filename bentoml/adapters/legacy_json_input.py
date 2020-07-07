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


class LegacyJsonInput(BaseInputAdapter):
    """LegacyJsonInput parses REST API request or CLI command into parsed_json(a
    dict in python) and pass down to user defined API function

    """

    BATCH_MODE_SUPPORTED = False

    def __init__(self, is_batch_input=False, **base_kwargs):
        super(LegacyJsonInput, self).__init__(
            is_batch_input=is_batch_input, **base_kwargs
        )

    def handle_request(self, request: flask.Request, func):
        if request.content_type.lower() == "application/json":
            parsed_json = json.loads(request.get_data(as_text=True))
        else:
            raise BadInput(
                "Request content-type must be 'application/json' for this "
                "BentoService API lambda endpoint"
            )

        result = func(parsed_json)
        return self.output_adapter.to_response(result, request)

    def handle_batch_request(
        self, requests: Iterable[SimpleRequest], func
    ) -> Iterable[SimpleResponse]:
        raise NotImplementedError("Use JsonInput instead to enable micro batching")

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
