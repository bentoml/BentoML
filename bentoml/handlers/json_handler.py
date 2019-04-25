# BentoML - Machine Learning Toolkit for packaging and deploying models
# Copyright (C) 2019 Atalaya Tech, Inc.

# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.

# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import json
import argparse

from flask import Response, make_response, jsonify

from bentoml.handlers.base_handlers import BentoHandler, get_output_str


class JsonHandler(BentoHandler):
    """JsonHandler parses REST API request or CLI command into parsed_json(a
    dict in python) and pass down to user defined API function
    """

    def handle_request(self, request, func):
        if request.content_type == 'application/json':
            parsed_json = json.loads(request.data.decode('utf-8'))
        else:
            return make_response(
                jsonify(message="Request content-type must be 'application/json'"
                        "for this BentoService API"), 400)

        result = func(parsed_json)
        result = get_output_str(result, request.headers.get('output', 'json'))
        return Response(response=result, status=200, mimetype="application/json")

    def handle_cli(self, args, func):
        parser = argparse.ArgumentParser()
        parser.add_argument('--input', required=True)
        parser.add_argument('-o', '--output', default="str", choices=['str', 'json', 'yaml'])
        parsed_args = parser.parse_args(args)

        if os.path.isfile(parsed_args.input):
            with open(parsed_args.input, 'r') as content_file:
                content = content_file.read()
        else:
            content = parsed_args.input

        input_json = json.loads(content)
        result = func(input_json)
        result = get_output_str(result, parsed_args.output)
        print(result)

    def handle_aws_lambda_event(self, event, func):
        if event['headers']['Content-Type'] == 'application/json':
            parsed_json = json.loads(event['body'])
        else:
            return {"statusCode": 400, "body": 'Only accept json as content type'}

        result = func(parsed_json)
        result = get_output_str(result, event['headers'].get('output', 'json'))
        return {"statusCode": 200, "body": result}
