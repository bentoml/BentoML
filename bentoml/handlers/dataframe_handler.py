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


def check_missing_columns(required_columns, df_columns):
    missing_columns = set(required_columns) - set(df_columns)
    if missing_columns:
        raise ValueError('Missing columns: {}'.format(','.join(missing_columns)))


class DataframeHandler(BentoHandler):
    """
    Create Data frame handler.  Dataframe handler will take inputs from rest request
    or cli options and return response for REST or stdout for CLI
    """

    def __init__(self, input_json_orient='records', output_json_orient='records',
                 input_columns_require=[]):
        self.input_json_orient = input_json_orient
        self.output_json_orient = output_json_orient
        self.input_columns_require = input_columns_require

    def handle_request(self, request, func):

        if request.headers.get('input_json_orient'):
            self.input_json_orient = request.headers['input_json_orient']

        if request.content_type == 'application/json':
            df = pd.read_json(
                request.data.decode('utf-8'), orient=self.input_json_orient, dtype=False)
        elif request.content_type == 'text/csv':
            df = pd.read_csv(request.data.decode('utf-8'))
        else:
            return make_response(400)

        if self.input_columns_require:
            check_missing_columns(self.input_columns_require, df.columns)

        output = func(df)

        if isinstance(output, pd.DataFrame):
            result = output.to_json(orient=self.output_json_orient)
        elif isinstance(output, np.ndarray):
            result = json.dumps(output.tolist())
        else:
            result = json.dumps(output)

        response = Response(response=result, status=200, mimetype="application/json")
        return response

    def handle_cli(self, args, func):
        parser = generate_cli_default_parser()
        parser.add_argument('--input_json_orient', default='records')
        parsed_args = parser.parse_args(args)

        # TODO: Add support for parsing cli argument string as input
        file_path = parsed_args.input

        if parsed_args.input_json_orient:
            self.input_json_orient = parsed_args.input_json_orient

        if file_path.endswith('.csv'):
            df = pd.read_csv(file_path)
        elif file_path.endswith('.json'):
            df = pd.read_json(file_path, orient=self.input_json_orient, dtype=False)
        else:
            raise ValueError("BentoML DataframeHandler currently only supports '.csv'"
                             "and '.json' files as cli input.")

        if self.input_columns_require:
            check_missing_columns(self.input_columns_require, df.columns)

        result = func(df)

        # TODO: revisit cli handler output format and options
        if isinstance(result, pd.DataFrame):
            print(result.to_json())
        elif isinstance(result, np.ndarray):
            print(json.dumps(result.tolist()))
        else:
            print(result)
