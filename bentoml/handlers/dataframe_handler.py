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
from bentoml.handlers.base_handlers import CliHandler, RequestHandler
from bentoml.handlers.utils import merge_dicts, generate_cli_default_parser


def check_missing_columns(required_columns, df_columns):
    missing_columns = set(required_columns) - set(df_columns)
    if missing_columns:
        raise ValueError('Missing columns: {}'.format(','.join(missing_columns)))


default_options = {
    'input_json_orient': 'records',
    'output_json_orient': 'records',
    'input_columns_require': []
}


class DataframeHandler(RequestHandler, CliHandler):
    """
    Create Data frame handler.  Dataframe handler will take inputs from rest request
    or cli options and return response for REST or stdout for CLI
    """

    @staticmethod
    def handle_request(request, func, options=None):
        options = merge_dicts(default_options, options)
        if request.headers.get('input_json_orient'):
            options['input_json_orient'] = request.headers['input_json_orient']

        if request.content_type == 'application/json':
            df = pd.read_json(
                request.data.decode('utf-8'), orient=options['input_json_orient'], dtype=False)
        elif request.content_type == 'text/csv':
            df = pd.read_csv(request.data.decode('utf-8'))
        else:
            return make_response(400)

        if options['input_columns_require']:
            check_missing_columns(options['input_columns_require'], df.columns)

        output = func(df)

        if isinstance(output, pd.DataFrame):
            result = output.to_json(orient=options['output_json_orient'])
        else:
            result = json.dumps(output)

        response = Response(response=result, status=200, mimetype="application/json")
        return response

    @staticmethod
    def handle_cli(args, func, options=None):
        parser = generate_cli_default_parser()
        parser.add_argument('--input_json_orient', default='records')
        parsed_args = parser.parse_args(args)
        options = merge_dicts(default_options, options)
        file_path = parsed_args.input

        if parsed_args.input_json_orient:
            options['input_json_orient'] = parsed_args.input_json_orient

        if file_path.endswith('.csv'):
            df = pd.read_csv(file_path)
        elif file_path.endswith('.json'):
            df = pd.read_json(file_path, orient=options['input_json_orient'], dtype=False)

        if options['input_columns_require']:
            check_missing_columns(options['input_columns_require'], df.columns)

        output = func(df)

        if parsed_args.output == 'json' or not parsed_args.output:
            if isinstance(output, pd.DataFrame):
                result = output.to_json(orient=options['output_json_orient'])
                result = json.loads(result)
                result = json.dumps(result, indent=2)
            else:
                result = json.dumps(output)
            sys.stdout.write(result)
        else:
            raise NotImplementedError
