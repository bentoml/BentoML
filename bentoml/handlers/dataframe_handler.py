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
import pandas as pd
from flask import Response, make_response
from bentoml.handlers import CliHandler, RequestHandler
from bentoml.cli.utils import generate_default_parser


class DataframeHandler(RequestHandler, CliHandler):
    """
    Create Data frame handler.  Dataframe handler will take inputs from rest request
    or cli options and return response for REST or stdout for CLI
    """

    @staticmethod
    def handle_request(request, func):
        if request.content_type == 'application/json':
            df = pd.read_json(request.data.decode('utf-8'))
        elif request.content_type == 'text/csv':
            df = pd.read_csv(request.data.decode('utf-8'))
        else:
            return make_response(400)

        output = func(df)

        if isinstance(output, pd.DataFrame):
            result = output.to_json(orient='records')
        else:
            result = json.dumps(output)

        response = Response(response=result, status=200, mimetype="application/json")
        return response

    @staticmethod
    def handle_cli(options, func):
        parser = generate_default_parser()
        parsed_args = parser.parse_args(options)

        with open(parsed_args.input, 'r') as content_file:
            content = content_file.read()
            if content_file.name.endswith('.csv'):
                df = pd.read_csv(content)
            elif content_file.name.endswith('.json'):
                df = pd.read_json(content)
            output = func(df)

            if parsed_args.output == 'json' or not parsed_args.output:
                if isinstance(output, pd.DataFrame):
                    result = output.to_json(orient='records')
                    result = json.loads(result)
                    result = json.dumps(result, indent=2)
                else:
                    result = json.dumps(output)
                sys.stdout.write(result)
            else:
                raise NotImplementedError
