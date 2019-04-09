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

import sys
import json
from flask import Response, make_response

from bentoml.handlers.base_handlers import RequestHandler, CliHandler
from bentoml.handlers.utils import merge_dicts, generate_cli_default_parser

default_options = {
    'input_json_orient': 'records',
    'output_json_orient': 'records',
}


class JsonHandler(RequestHandler, CliHandler):
    """
    Json handler take input json str and process them and return response or stdout.
    """

    @staticmethod
    def handle_request(request, func, options=None):
        options = merge_dicts(default_options, options)
        if request.content_type == 'application/json':
            parsed_json = json.loads(request.data.decode('utf-8'))
        else:
            return make_response(400)

        output = func(parsed_json)
        try:
            result = json.dumps(output)
        except Exception as e:  # pylint:disable=W0703
            if isinstance(e, TypeError):
                if type(output).__module__ == 'numpy':
                    output = output.tolist()
                    result = json.dumps(output)
                else:
                    raise e
            else:
                raise e

        response = Response(response=result, status=200, mimetype="application/json")
        return response

    @staticmethod
    def handle_cli(args, func, options=None):
        options = merge_dicts(default_options, options)
        parser = generate_cli_default_parser()
        parsed_args = parser.parse_args(args)

        with open(parsed_args.input, 'r') as content_file:
            content = content_file.read()
            input_json = json.loads(content)
            output = func(input_json)

            try:
                result = json.dumps(output)
            except Exception as e:  # pylint:disable=W0703
                if isinstance(e, TypeError):
                    if type(output).__module__ == 'numpy':
                        output = output.tolist()
                        result = json.dumps(output)
                    else:
                        raise e
                else:
                    raise e

            if parsed_args.output == 'json' or not parsed_args.output:
                try:
                    sys.stdout.write(result)
                except Exception as e:
                    raise e
            else:
                raise NotImplementedError
