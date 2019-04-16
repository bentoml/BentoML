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

import numpy as np
import pandas as pd
from werkzeug.utils import secure_filename
from flask import Response

from bentoml.handlers.base_handlers import BentoHandler
from bentoml.handlers.utils import generate_cli_default_parser


def check_file_format(file_name, accept_format_list):
    """
    Raise error if file's extension is not in the accept_format_list
    """
    if accept_format_list:
        _, extension = os.path.splitext(file_name)
        if extension not in accept_format_list:
            raise ValueError('File format does not include in the white list')


class ImageHandler(BentoHandler):
    """
    Image handler take input image and process them and return response or stdout.
    """

    def __init__(self, input_names=["image"], accept_multiple_files=False,
                 accept_file_extensions=['.jpg', '.png', '.jpeg'], output_format='json'):
        self.input_names = input_names
        self.accept_multiple_files = accept_multiple_files
        self.accept_file_extensions = accept_file_extensions
        self.output_format = output_format

    def handle_request(self, request, func):
        """
        Handle http request that has image file/s.  It will convert image into a
        ndarray for the function to consume.

        Args:
            request: incoming request object.
            func: function that will take ndarray as its arg.
            options: configuration for handling request object.
        Return:
            response object
        """

        if request.method == 'POST':
            if not self.accept_multiple_files:
                input_file = request.files[self.input_names[0]]
                file_name = secure_filename(input_file.filename)

                check_file_format(file_name, self.accept_file_extensions)

                input_data_string = input_file.read()
                input_data = np.fromstring(input_data_string, np.uint8)
            else:
                return Response(response="Only support single file input", status=400)

            output = func(input_data)

            if self.output_format == 'json':
                result = json.dumps(output)
                response = Response(response=result, status=200, mimetype='applications/json')
                return response
            else:
                return Response(response="Only support json output", status=400)
        else:
            return Response(response="Only accept POST request", status=400)

    def handle_cli(self, args, func):
        parser = generate_cli_default_parser()
        parsed_args = parser.parse_args(args)
        file_path = parsed_args.input

        check_file_format(file_path, self.accept_file_extensions)
        if not os.path.isabs(file_path):
            file_path = os.path.abspath(file_path)

        try:
            import cv2
        except ImportError:
            raise ImportError("opencv-python package is required to use ImageHandler")

        image = cv2.imread(file_path)

        result = func(image)
        # TODO: revisit cli handler output format and options
        if isinstance(result, pd.DataFrame):
            print(result.to_json())
        elif isinstance(result, np.ndarray):
            print(json.dumps(result.tolist()))
        else:
            print(result)
