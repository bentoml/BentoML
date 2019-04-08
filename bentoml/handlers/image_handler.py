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
import os
import sys
from werkzeug.utils import secure_filename
from flask import request, Response, make_response
from bentoml.handlers.base_handlers import RequestHandler, CliHandler
from bentoml.handlers.utils import merge_dicts, generate_cli_default_parser

default_options = {
    'input_names': ['image'],
    'accept_multiply_files': False,
    'accept_file_extensions': ['.jpg', '.png', '.jpeg']
}


def check_file_format(file_name, accept_format_list):
    """
    Raise error if file's extension is not in the accept_format_list
    """
    if accept_format_list:
        name, extension = os.path.splitext(file_name)
        if extension not in accept_format_list:
            raise ValueError('File format does not include in the white list')


class ImageHandler(RequestHandler, CliHandler):
    """
    Image handler take input image and process them and return response or stdout.
    """

    @staticmethod
    def handle_request(request, func, options={}):
        if request.method == 'POST':
            try:
                options = merge_dicts(default_options, options)
                if not options.accept_multiply_files:
                    input_file = request.files[options.input_name]
                    file_name = secure_filename(input_file.filename)

                    check_file_format(file_name, options.accept_file_extensions) 

                    input_data = input_file.read()
                else:
                    raise NotImplementedError

                output = func(input_data)

                result = json.dumps(output)
                response = Response(response=result, status=200, mimetype='applications/json')
                return response
            except Exception as e:
                # TODO: handle exceptions
                return make_response(500)
        else:
            return make_response(500)

    @staticmethod
    def handle_cli(args, func, options=None):
        options = merge_dicts(default_options, options)
        parser = generate_cli_default_parser()
        parsed_args = parser.parse_args(args)

        try:
            check_file_format(parsed_args.input, options.accept_format_list)
            with open(parsed_args.input, 'rb') as content_file:
                content = content_file.read()
                output = func(content)

                if parsed_args.output == 'json' or not parsed_args.output:
                    result = json.dumps(output)
                    sys.stdout.write(result)
                else:
                    raise NotImplementedError
        except Exception as e:
            raise NotImplementedError
