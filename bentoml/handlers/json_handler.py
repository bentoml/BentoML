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

import json
import pandas as pd
import numpy as np
from flask import Response, make_response

from bentoml.handlers.base_handlers import BentoHandler
from bentoml.handlers.utils import generate_cli_default_parser


class JsonHandler(BentoHandler):
    """
    Json handler take input json str and process them and return response or stdout.
    """

    def __init__(self, input_json_orient='records', output_json_orient='records'):
        self.input_json_orient = input_json_orient
        self.output_json_orient = output_json_orient

    def handle_request(self, request, func):
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

    def handle_cli(self, args, func):
        parser = generate_cli_default_parser()
        parsed_args = parser.parse_args(args)

        with open(parsed_args.input, 'r') as content_file:
            content = content_file.read()

        input_json = json.loads(content)
        result = func(input_json)

        # TODO: revisit cli handler output format and options
        if isinstance(result, pd.DataFrame):
            print(result.to_json())
        elif isinstance(result, np.ndarray):
            print(json.dumps(result.tolist()))
        else:
            print(result)
