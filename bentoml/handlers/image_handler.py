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
import argparse

import numpy as np
from werkzeug.utils import secure_filename
from flask import Response

from bentoml.handlers.base_handlers import BentoHandler, get_output_str


def check_file_format(file_name, accept_format_list):
    """
    Raise error if file's extension is not in the accept_format_list
    """
    if accept_format_list:
        _, extension = os.path.splitext(file_name)
        if extension not in accept_format_list:
            raise ValueError(
                'Input file not in supported format list: {}'.format(accept_format_list))


class ImageHandler(BentoHandler):
    """Image handler take input image and process them and return response or stdout.
    """

    def __init__(self, input_names=None, accept_file_extensions=None, accept_multiple_files=False):
        self.input_names = input_names or ["image"]
        self.accept_file_extensions = accept_file_extensions or ['.jpg', '.png', '.jpeg']
        self.accept_multiple_files = accept_multiple_files

    def handle_request(self, request, func):
        """Handle http request that has image file/s.  It will convert image into a
        ndarray for the function to consume.

        Args:
            request: incoming request object.
            func: function that will take ndarray as its arg.
            options: configuration for handling request object.
        Return:
            response object
        """

        if request.method != 'POST':
            return Response(response="Only accept POST request", status=400)

        if not self.accept_multiple_files:
            input_file = request.files[self.input_names[0]]
            file_name = secure_filename(input_file.filename)

            check_file_format(file_name, self.accept_file_extensions)

            input_data_string = input_file.read()
            input_data = np.fromstring(input_data_string, np.uint8)
        else:
            return Response(response="Only support single file input", status=400)

        result = func(input_data)
        result = get_output_str(result, request.headers.get('output', 'json'))
        return Response(response=result, status=200, mimetype="application/json")

    def handle_cli(self, args, func):
        parser = argparse.ArgumentParser()
        parser.add_argument('--input', required=True)
        parser.add_argument('-o', '--output', default="str", choices=['str', 'json', 'yaml'])
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
        result = get_output_str(result, output_format=parsed_args.output)
        print(result)

    def handle_aws_lambda_event(self, event, func):
        raise NotImplementedError
