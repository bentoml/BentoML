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
from bentoml.exceptions import BentoMLException
from bentoml.handlers.base_handlers import BentoHandler, get_output_str

try:
    from imageio import imread
except ImportError:
    imread = None


def verify_image_format_or_raise(file_name, accept_format_list):
    """
    Raise error if file's extension is not in the accept_format_list
    """
    if accept_format_list:
        _, extension = os.path.splitext(file_name)
        if extension not in accept_format_list:
            raise ValueError(
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

    def __init__(
        self, input_names=("image",), accept_image_formats=None, pilmode="RGB"
    ):
        if imread is None:
            raise ImportError("imageio package is required to use ImageHandler")

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
        if request.method != "POST":
            return Response(response="Only accept POST request", status=400)

        input_files = [request.files.get(filename) for filename in self.input_names]
        if len(input_files) == 1 and input_files[0] is None:
            data = request.get_data()
            if data:
                input_streams = (data,)
            else:
                raise ValueError(
                    "BentoML#ImageHandler unexpected HTTP request: %s" % request
                )
        else:
            file_names = [secure_filename(file.filename) for file in input_files]
            for file_name in file_names:
                verify_image_format_or_raise(file_name, self.accept_image_formats)
            input_streams = [BytesIO(input_file.read()) for input_file in input_files]

        input_data = tuple(
            imread(input_stream, pilmode=self.pilmode) for input_stream in input_streams
        )
        result = func(*input_data)

        result = get_output_str(result, request.headers.get("output", "json"))
        return Response(response=result, status=200, mimetype="application/json")

    def handle_cli(self, args, func):
        parser = argparse.ArgumentParser()
        parser.add_argument("--input", required=True)
        parser.add_argument(
            "-o", "--output", default="str", choices=["str", "json", "yaml"]
        )
        parsed_args = parser.parse_args(args)
        file_path = parsed_args.input

        verify_image_format_or_raise(file_path, self.accept_image_formats)
        if not os.path.isabs(file_path):
            file_path = os.path.abspath(file_path)

        image_array = imread(file_path, pilmode=self.pilmode)

        result = func(image_array)
        result = get_output_str(result, output_format=parsed_args.output)
        print(result)

    def handle_aws_lambda_event(self, event, func):
        if event["headers"].get("Content-Type", "").startswith("images/"):
            # decodebytes introduced at python3.1
            try:
                image = imread(base64.decodebytes(event["body"]), pilmode=self.pilmode)
            except AttributeError:
                image = imread(
                    base64.decodestring(event["body"]),  # pylint: disable=W1505
                    pilmode=self.pilmode,
                )
        else:
            raise BentoMLException(
                "BentoML currently doesn't support Content-Type: {content_type} for "
                "AWS Lambda".format(content_type=event["headers"]["Content-Type"])
            )

        result = func(image)
        result = get_output_str(result, event["headers"].get("output", "json"))
        return {"statusCode": 200, "body": result}

    def handle_clipper_bytes(self, inputs, func):
        try:
            import cv2
        except ImportError:
            raise ImportError("opencv-python package is required to use ImageHandler")

        def transform_and_predict(input_bytes):
            data = cv2.imdecode(input_bytes, cv2.IMREAD_COLOR)
            return func(data)

        return list(map(transform_and_predict, inputs))

    def handle_clipper_strings(self, inputs, func):
        raise RuntimeError(
            "ImageHandler does not support 'strings' input_type \
                for Clipper deployment at the moment"
        )

    def handle_clipper_ints(self, inputs, func):
        raise RuntimeError(
            "ImageHandler doesn't support ints input types \
                for clipper deployment at the moment"
        )

    def handle_clipper_doubles(self, inputs, func):
        raise RuntimeError(
            "ImageHandler doesn't support doubles input types \
                for clipper deployment at the moment"
        )

    def handle_clipper_floats(self, inputs, func):
        raise RuntimeError(
            "ImageHandler doesn't support floats input types \
                for clipper deployment at the moment"
        )
