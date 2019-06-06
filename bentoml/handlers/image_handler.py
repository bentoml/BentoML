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

import numpy as np
from werkzeug.utils import secure_filename
from flask import Response

from bentoml.utils.exceptions import BentoMLException
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
    """Image handler take input image and process them and return response or stdout.
    """

    def __init__(
        self, input_names=None, accept_file_extensions=None, accept_multiple_files=False
    ):
        self.input_names = input_names or ["image"]
        self.accept_file_extensions = accept_file_extensions or [
            ".jpg",
            ".png",
            ".jpeg",
        ]
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

        if request.method != "POST":
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
            import cv2
        except ImportError:
            raise ImportError("opencv-python package is required to use ImageHandler")

        image = cv2.imread(file_path)
        result = func(image)
        result = get_output_str(result, output_format=parsed_args.output)
        print(result)

    def handle_aws_lambda_event(self, event, func):
        try:
            import cv2
        except ImportError:
            raise ImportError("opencv-python package is required to use ImageHandler")

        if event["headers"].get("Content-Type", None) in ACCEPTED_CONTENT_TYPES:
            nparr = np.fromstring(base64.b64decode(event["body"]), np.uint8)
            image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        else:
            raise BentoMLException(
                "BentoML currently doesn't support Content-Type: {content_type} for AWS Lambda".format(
                    content_type=event["headers"]["Content-Type"]
                )
            )

        result = func(image)
        result = get_output_str(result, event["headers"].get("output", "json"))
        return {"statusCode": 200, "body": result}
