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

from werkzeug.exceptions import BadRequest
from flask import Response


from bentoml.handlers.base_handlers import BentoHandler, get_output_str


class JsonHandler(BentoHandler):
    """JsonHandler parses REST API request or CLI command into parsed_json(a
    dict in python) and pass down to user defined API function

    """

    def handle_request(self, request, func):
        if request.content_type == "application/json":
            parsed_json = json.loads(request.data.decode("utf-8"))
        else:
            raise BadRequest(
                "Request content-type must be 'application/json' for this "
                "BentoService API"
            )

        result = func(parsed_json)
        result = get_output_str(result, request.headers.get("output", "json"))
        return Response(response=result, status=200, mimetype="application/json")

    def handle_cli(self, args, func):
        parser = argparse.ArgumentParser()
        parser.add_argument("--input", required=True)
        parser.add_argument(
            "-o", "--output", default="str", choices=["str", "json", "yaml"]
        )
        parsed_args = parser.parse_args(args)

        if os.path.isfile(parsed_args.input):
            with open(parsed_args.input, "r") as content_file:
                content = content_file.read()
        else:
            content = parsed_args.input

        input_json = json.loads(content)
        result = func(input_json)
        result = get_output_str(result, parsed_args.output)
        print(result)

    def handle_aws_lambda_event(self, event, func):
        if event["headers"]["Content-Type"] == "application/json":
            parsed_json = json.loads(event["body"])
        else:
            raise BadRequest("Only accept json as content type")

        result = func(parsed_json)
        result = get_output_str(result, event["headers"].get("output", "json"))
        return {"statusCode": 200, "body": result}
