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
import numpy as np
from werkzeug.utils import secure_filename
from flask import request, Response, make_response
from bentoml.handlers.base_handlers import RequestHandler, CliHandler
from bentoml.handlers.utils import merge_dicts, generate_cli_default_parser

default_options = {
    'input_names': ['image'],
    'accept_multiply_files': False,
    'accept_file_extensions': ['.jpg', '.png', '.jpeg'],
    'output_format': 'json'
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
        """
        Handle http request that has image file/s.  It will convert image into a ndarray for the function to consume.

        Args:
            request: incoming request object.
            func: function that will take ndarray as its arg.
            options: configuration for handling request object.
        Return:
            response object
        """

        if request.method == 'POST':
            try:
                options = merge_dicts(default_options, options)
                if not options.accept_multiply_files:
                    input_file = request.files[options.input_name]
                    file_name = secure_filename(input_file.filename)

                    check_file_format(file_name, options['accept_file_extensions'])

                    input_data_string = input_file.read()
                    input_data = np.fromstring(input_data_string, np.uint8)
                else:
                    raise NotImplementedError

                output = func(input_data)

                if options['output_format'] == 'json':
                    result = json.dumps(output)
                    response = Response(response=result, status=200, mimetype='applications/json')
                    return response
                else:
                    raise NotImplementedError
            except Exception as e:
                # TODO: handle exceptions
                return make_response(500)
        else:
            return make_response(500)

    @staticmethod
    def handle_cli(args, func, options={}):
        options = merge_dicts(default_options, options)
        parser = generate_cli_default_parser()
        parsed_args = parser.parse_args(args)
        file_path = parsed_args.input

        try:
            check_file_format(file_path, options['accept_file_extensions'])
            if not os.path.isabs(file_path):
                file_path = os.path.abspath(file_path)

            try:
                import cv2
            except ImportError:
                raise ImportError("opencv-python package is required to use ImageHandler")

            image = cv2.imread(file_path)
            output = func(image)

            if options['output_format'] == 'json':
                result = json.dumps(output)
                sys.stdout.write(result)
            else:
                raise NotImplementedError

        except Exception as e:
            # TODO: handle exceptions
            raise NotImplementedError
