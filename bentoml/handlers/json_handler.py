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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import json
import argparse
from typing import Iterable

from flask import Response

from bentoml.exceptions import BadInput
from bentoml.marshal.utils import SimpleResponse, SimpleRequest
from bentoml.handlers.base_handlers import BentoHandler, api_func_result_to_json
from bentoml.handlers.utils import concat_list


class BadResult:
    pass


class JsonHandler(BentoHandler):
    """JsonHandler parses REST API request or CLI command into parsed_json(a
    dict in python) and pass down to user defined API function

    """

    BATCH_MODE_SUPPORTED = True

    def __init__(self, is_batch_input=False, **base_kwargs):
        super(JsonHandler, self).__init__(is_batch_input=is_batch_input, **base_kwargs)

    def handle_request(self, request, func):
        if request.content_type == "application/json":
            parsed_json = json.loads(request.data.decode("utf-8"))
        else:
            raise BadInput(
                "Request content-type must be 'application/json' for this "
                "BentoService API"
            )

        result = func(parsed_json)
        json_output = api_func_result_to_json(result)
        return Response(response=json_output, status=200, mimetype="application/json")

    def handle_batch_request(
        self, requests: Iterable[SimpleRequest], func
    ) -> Iterable[SimpleResponse]:
        bad_resp = SimpleResponse(400, None, "Bad Input")
        instances_list = [None] * len(requests)
        responses = [bad_resp] * len(requests)
        batch_flags = [None] * len(requests)

        for i, request in enumerate(requests):
            batch_flags[i] = (
                request.formated_headers.get(
                    self._BATCH_REQUEST_HEADER.lower(),
                    "true" if self.config.get('is_batch_input') else "false",
                )
                == "true"
            )
            try:
                raw_str = request.data
                parsed_json = json.loads(raw_str)
                if not batch_flags[i]:
                    parsed_json = (parsed_json,)
                instances_list[i] = parsed_json
            except (json.JSONDecodeError, UnicodeDecodeError):
                responses[i] = SimpleResponse(400, None, "Not a valid json")
            except Exception:  # pylint: disable=broad-except
                import traceback

                err = traceback.format_exc()
                responses[i] = SimpleResponse(
                    500, None, f"Internal Server Error: {err}"
                )

        merged_instances, slices = concat_list(instances_list)
        merged_result = func(merged_instances)
        if not isinstance(merged_result, (list, tuple)) or len(merged_result) != len(
            merged_instances
        ):
            raise ValueError(
                "The return value with JsonHandler must be list of jsonable objects, "
                "and have same length as the inputs."
            )

        for i, s in enumerate(slices):
            if s is None:
                continue
            result = merged_result[s]
            if not batch_flags[i]:
                result = result[0]
            result_str = api_func_result_to_json(result)
            responses[i] = SimpleResponse(200, dict(), result_str)

        return responses

    def handle_cli(self, args, func):
        parser = argparse.ArgumentParser()
        parser.add_argument("--input", required=True)
        parser.add_argument("-o", "--output", default="str", choices=["str", "json"])
        parsed_args = parser.parse_args(args)

        if os.path.isfile(parsed_args.input):
            with open(parsed_args.input, "r") as content_file:
                content = content_file.read()
        else:
            content = parsed_args.input

        input_json = json.loads(content)
        result = func(input_json)
        if parsed_args.output == 'json':
            result = api_func_result_to_json(result)
        else:
            result = str(result)
        print(result)

    def handle_aws_lambda_event(self, event, func):
        if event["headers"]["Content-Type"] == "application/json":
            parsed_json = json.loads(event["body"])
        else:
            raise BadInput(
                "Request content-type must be 'application/json' for this "
                "BentoService API lambda endpoint"
            )

        result = func(parsed_json)
        json_output = api_func_result_to_json(result)
        return {"statusCode": 200, "body": json_output}
