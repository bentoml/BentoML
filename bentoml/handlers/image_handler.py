# Copyright 2019 Atalaya Tech, Inc.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

# http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import argparse
import base64
from io import BytesIO

from werkzeug.utils import secure_filename
from flask import Response

from bentoml import config
from bentoml.exceptions import BadInput, MissingDependencyException
from bentoml.handlers.base_handlers import BentoHandler, api_func_result_to_json


def _import_imageio_imread():
    try:
        from imageio import imread
    except ImportError:
        raise MissingDependencyException(
            "imageio package is required to use ImageHandler"
        )

    return imread


def verify_image_format_or_raise(file_name, accept_format_list):
    """
    Raise error if file's extension is not in the accept_format_list
    """
    if accept_format_list:
        _, extension = os.path.splitext(file_name)
        if extension.lower() not in accept_format_list:
            raise BadInput(
                "Input file not in supported format list: {}".format(accept_format_list)
            )


def get_default_accept_image_formats():
    """With default bentoML config, this returns:
        ['.jpg', '.png', '.jpeg', '.tiff', '.webp', '.bmp']
    """
    return [
        extension.strip()
        for extension in config('apiserver')
        .get('default_image_handler_accept_file_extensions')
        .split(',')
    ]


class ImageHandler(BentoHandler):
    """Transform incoming image data from http request, cli or lambda event into numpy
    array.

    Handle incoming image data from different sources, transform them into numpy array
    and pass down to user defined API functions

    Args:
        input_names (string[]]): A tuple of acceptable input name for HTTP request.
            Default value is (image,)
        accept_image_formats (string[]):  A list of acceptable image formats.
            Default value is loaded from bentoml config
            'apiserver/default_image_handler_accept_file_extensions', which is
            set to ['.jpg', '.png', '.jpeg', '.tiff', '.webp', '.bmp'] by default.
            List of all supported format can be found here:
            https://imageio.readthedocs.io/en/stable/formats.html
        pilmode (string): The pilmode to be used for reading image file into numpy
            array. Default value is 'RGB'.  Find more information at:
            https://imageio.readthedocs.io/en/stable/format_png-pil.html

    Raises:
        ImportError: imageio package is required to use ImageHandler
    """

    HTTP_METHODS = ["POST"]

    def __init__(
        self, input_names=("image",), accept_image_formats=None, pilmode="RGB"
    ):
        self.imread = _import_imageio_imread()

        self.input_names = tuple(input_names)
        self.pilmode = pilmode
        self.accept_image_formats = (
            accept_image_formats or get_default_accept_image_formats()
        )

    @property
    def request_schema(self):
        return {
            "image/*": {"schema": {"type": "string", "format": "binary"}},
            "multipart/form-data": {
                "schema": {
                    "type": "object",
                    "properties": {
                        filename: {"type": "string", "format": "binary"}
                        for filename in self.input_names
                    },
                }
            },
        }

    @property
    def pip_dependencies(self):
        return ['imageio']

    def handle_request(self, request, func):
        """Handle http request that has image file/s. It will convert image into a
        ndarray for the function to consume.

        Args:
            request: incoming request object.
            func: function that will take ndarray as its arg.
            options: configuration for handling request object.
        Return:
            response object
        """
        if len(self.input_names) == 1 and len(request.files) == 1:
            # Ignore multipart form input name when ImageHandler is intended to accept
            # only one image file at a time
            input_files = [file for _, file in request.files.items()]
        else:
            input_files = [
                request.files.get(form_input_name)
                for form_input_name in self.input_names
                if form_input_name in request.files
            ]

        if input_files:
            file_names = [secure_filename(file.filename) for file in input_files]
            for file_name in file_names:
                verify_image_format_or_raise(file_name, self.accept_image_formats)
            input_streams = [BytesIO(input_file.read()) for input_file in input_files]
        else:
            data = request.get_data()
            if data:
                input_streams = (data,)
            else:
                raise BadInput("BentoML#ImageHandler unexpected HTTP request format")

        input_data = tuple(
            self.imread(input_stream, pilmode=self.pilmode)
            for input_stream in input_streams
        )
        result = func(*input_data)
        json_output = api_func_result_to_json(result)
        return Response(response=json_output, status=200, mimetype="application/json")

    def handle_cli(self, args, func):
        parser = argparse.ArgumentParser()
        parser.add_argument("--input", required=True)
        parser.add_argument("-o", "--output", default="str", choices=["str", "json"])
        parsed_args = parser.parse_args(args)
        file_path = parsed_args.input

        verify_image_format_or_raise(file_path, self.accept_image_formats)
        if not os.path.isabs(file_path):
            file_path = os.path.abspath(file_path)

        image_array = self.imread(file_path, pilmode=self.pilmode)

        result = func(image_array)
        if parsed_args.output == "json":
            result = api_func_result_to_json(result)
        else:
            result = str(result)
        print(result)

    def handle_aws_lambda_event(self, event, func):
        if event["headers"].get("Content-Type", "").startswith("images/"):
            image = self.imread(base64.decodebytes(event["body"]), pilmode=self.pilmode)
        else:
            raise BadInput(
                "BentoML currently doesn't support Content-Type: {content_type} for "
                "AWS Lambda".format(content_type=event["headers"]["Content-Type"])
            )

        result = func(image)
        json_output = api_func_result_to_json(result)
        return {"statusCode": 200, "body": json_output}
