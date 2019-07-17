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

from bentoml.exceptions import BentoMLException
from bentoml.handlers.base_handlers import BentoHandler, get_output_str

ACCEPTED_CONTENT_TYPES = ["images/png", "images/jpeg", "images/jpg"]


def check_file_format(file_name, accept_format_list):
    """
    Raise error if file's extension is not in the accept_format_list
    """
    if accept_format_list:
        _, extension = os.path.splitext(file_name)
        if extension not in accept_format_list:
            raise ValueError(
                "Input file not in supported format list: {}".format(accept_format_list)
            )


class ImageHandler(BentoHandler):
    """Transform incoming image data from http request, cli or lambda event into numpy
    array.

    Handle incoming image data from different sources, transform them into numpy array
    and pass down to user defined API functions

    Args:
        input_name (string[]]): A list of acceptable input name for HTTP request.
            Default value is image
        accept_file_extensions (string[]):  A list of acceptable image extensions.
            Default value is [.jpg, .jpeg, .png]
        accept_multiple_files (boolean):  Accept multiple files in single request or
            not. Default value is False
        pilmode (string): The pilmode to be used for reading image file into numpy
            array. Default value is RGB.  Find more information at
            https://imageio.readthedocs.io/en/stable/format_png-pil.html#png-pil

    Raises:
        ImportError: imageio package is required to use ImageHandler
    """

    def __init__(
        self,
        input_name="image",
        accept_file_extensions=None,
        accept_multiple_files=False,
        pilmode="RGB",
    ):
        self.input_name = input_name
        self.pilmode = pilmode
        self.accept_file_extensions = accept_file_extensions or [
            ".jpg",
            ".png",
            ".jpeg",
        ]
        self.accept_multiple_files = accept_multiple_files

    @property
    def request_schema(self):
        return {
            "image/*": {"schema": {"type": "string", "format": "binary"}},
            "multipart/form-data": {
                "schema": {"type": "object"},
                "properties": {self.input_name: {"type": "string", "format": "binary"}},
            },
        }

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
        try:
            from imageio import imread
        except ImportError:
            raise ImportError("imageio package is required to use ImageHandler")

        if request.method != "POST":
            return Response(response="Only accept POST request", status=400)

        if not self.accept_multiple_files:
            input_file = request.files.get("image")

            if input_file:
                file_name = secure_filename(input_file.filename)
                check_file_format(file_name, self.accept_file_extensions)
                input_stream = BytesIO(input_file.read())
            elif request.data:
                input_stream = request.data
            else:
                raise ValueError(
                    "BentoML#ImageHandler unexpected HTTP request: %s" % request
                )

            input_data = imread(input_stream, pilmode=self.pilmode)
        else:
            return Response(response="Only support single file input", status=400)

        result = func(input_data)
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

        check_file_format(file_path, self.accept_file_extensions)
        if not os.path.isabs(file_path):
            file_path = os.path.abspath(file_path)

        try:
            from imageio import imread
        except ImportError:
            raise ImportError("imageio package is required to use ImageHandler")

        image_array = imread(file_path, pilmode=self.pilmode)

        result = func(image_array)
        result = get_output_str(result, output_format=parsed_args.output)
        print(result)

    def handle_aws_lambda_event(self, event, func):
        try:
            from imageio import imread
        except ImportError:
            raise ImportError("imageio package is required to use ImageHandler")

        if event["headers"].get("Content-Type", None) in ACCEPTED_CONTENT_TYPES:
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
