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
import numpy as np

from bentoml.exceptions import BentoMLException
from bentoml.handlers.base_handlers import BentoHandler, get_output_str
from bentoml.handlers.image_handler import ACCEPTED_CONTENT_TYPES, check_file_format


class FastaiImageHandler(BentoHandler):
    """Transform incoming image data to fastai.vision.Image

    Handle incoming image data, process them into fastai.vision.Image instance and
    pass down to user defined API functions


    Args:
        input_name ([str]]): A list of acceptable input name for HTTP request.
            Default value is image
        accept_file_extensions ([str]):  A list of acceptable image extensions.
            Default value is [.jpg, .jpeg, .png]
        accept_multiple_files (boolean):  Accept multiple files in single request or
            not. Default value is False
        convert_mode (str): The pilmode to be used for reading image file into
            numpy array. Default value is RGB.  Find more information at
            https://imageio.readthedocs.io/en/stable/format_png-pil.html#png-pil
        div (bool): If True, pixel values are divided by 255 to become floats
            between 0. and 1.
        cls (Class): Parameter from fastai.vision ``open_image``, default is
            ``fastai.vision.Image``
        after_open (func): Parameter from fastai.vision ``open_image``, default
            is None

    Raises:
        ImportError: imageio package is required to use FastaiImageHandler
        ImportError: fastai package is required to use FastaiImageHandler
    """

    def __init__(
        self,
        input_name=None,
        accept_file_extensions=None,
        convert_mode=None,
        div=True,
        cls=None,
        after_open=None,
    ):
        self.input_name = input_name or "image"
        self.convert_mode = convert_mode or "RGB"
        self.div = (div or True,)
        self.cls = cls
        self.accept_file_extensions = accept_file_extensions or [".jpg", ".png", "jpeg"]
        self.after_open = after_open

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
        try:
            from fastai.vision import Image, pil2tensor
        except ImportError:
            raise ImportError("fastai package is required to use FastaiImageHandler")

        try:
            from imageio import imread
        except ImportError:
            raise ImportError("imageio package is required to use FastaiImageHandler")

        if request.method != "POST":
            return Response(response="Only accept POST request", status=400)

        input_file = request.files.get(self.input_name)

        if input_file:
            file_name = secure_filename(input_file.filename)
            check_file_format(file_name, self.accept_file_extensions)
            input_stream = BytesIO(input_file.read())
        elif request.data:
            input_stream = request.data
        else:
            raise ValueError(
                "BentoML#FastaiImageHandler unexpected HTTP request: %s" % request
            )

        input_data = imread(input_stream, pilmode=self.convert_mode)

        if self.after_open:
            input_data = self.after_open(input_data)

        input_data = pil2tensor(input_data, np.float32)

        if self.div:
            input_data = input_data.div_(255)

        if self.cls:
            input_data = self.cls(input_data)
        else:
            input_data = Image(input_data)

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
            from fastai.vision import open_image, Image
        except ImportError:
            raise ImportError("fastai package is required to use")

        image_array = open_image(
            fn=file_path,
            convert_mode=self.convert_mode,
            div=self.div,
            after_open=self.after_open,
            cls=self.cls or Image,
        )

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
                image_data = imread(
                    base64.decodebytes(event["body"]), pilmode=self.pilmode
                )
            except AttributeError:
                image_data = imread(
                    base64.decodestring(event["body"]),  # pylint: disable=W1505
                    pilmode=self.convert_mode,
                )
        else:
            raise BentoMLException(
                "BentoML currently doesn't support Content-Type: {content_type} for "
                "AWS Lambda".format(content_type=event["headers"]["Content-Type"])
            )
        try:
            from fastai.vision import pil2tensor, Image
        except ImportError:
            raise ImportError("fastai package is required")

        if self.after_open:
            image_data = self.after_open(image_data)

        image_data = pil2tensor(image_data, np.float32)
        if self.div:
            image_data = image_data.div_(255)
        if self.cls:
            image_data = self.cls(image_data)
        else:
            image_data = Image(image_data)

        result = func(image_data)
        result = get_output_str(result, event["headers"].get("output", "json"))
        return {"statusCode": 200, "body": result}

    def handle_clipper_bytes(self, inputs, func):
        try:
            import cv2
        except ImportError:
            raise ImportError(
                "opencv-python package is required to use FastaiImageHandler"
            )

        try:
            from fastai.vision import pil2tensor, Image
            import numpy as np
        except ImportError:
            raise ImportError("fastai package is required to use")

        def transform_and_predict(input_bytes):
            image_data = cv2.imdecode(input_bytes, cv2.IMREAD_COLOR)
            if self.after_open:
                image_data = self.after_open(image_data)
            image_data = pil2tensor(image_data, np.float32)

            if self.div:
                image_data = image_data.div_(255)

            if self.cls:
                image_data = self.cls(image_data)
            else:
                image_data = Image(image_data)

            return func(image_data)

        return list(map(transform_and_predict, inputs))

    def handle_clipper_strings(self, inputs, func):
        raise RuntimeError(
            "Fastai Image handler does not support 'strings' input_type \
                for Clipper deployment at the moment"
        )

    def handle_clipper_ints(self, inputs, func):
        raise RuntimeError(
            "Fastai ImageHandler doesn't support ints input types \
                for clipper deployment at the moment"
        )

    def handle_clipper_doubles(self, inputs, func):
        raise RuntimeError(
            "Fastai ImageHandler doesn't support doubles input types \
                for clipper deployment at the moment"
        )

    def handle_clipper_floats(self, inputs, func):
        raise RuntimeError(
            "Fastai ImageHandler doesn't support floats input types \
                for clipper deployment at the moment"
        )
