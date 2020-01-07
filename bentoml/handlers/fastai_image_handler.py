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

from bentoml.exceptions import BadInput, MissingDependencyException
from bentoml.handlers.base_handlers import BentoHandler, api_func_result_to_json
from bentoml.handlers.image_handler import (
    verify_image_format_or_raise,
    get_default_accept_image_formats,
)


def _import_fastai_vision():
    try:
        from fastai import vision
    except ImportError:
        raise MissingDependencyException(
            "fastai.vision package is required to use FastaiImageHandler"
        )

    return vision


def _import_imageio_imread():
    try:
        from imageio import imread
    except ImportError:
        raise MissingDependencyException(
            "imageio package is required to use FastaiImageHandler"
        )

    return imread


class FastaiImageHandler(BentoHandler):
    """BentoHandler specified for handling image input following fastai conventions
    by passing type fastai.vision.Image to user API function and providing options
    such as div, cls, and after_open

    Args:
        input_names ([str]]): A tuple of acceptable input name for HTTP request.
            Default value is (image,)
        accept_image_formats ([str]):  A list of acceptable image formats.
            Default value is loaded from bentoml config
            'apiserver/default_image_handler_accept_file_extensions', which is
            set to ['.jpg', '.png', '.jpeg', '.tiff', '.webp', '.bmp'] by default.
            List of all supported format can be found here:
            https://imageio.readthedocs.io/en/stable/formats.html
        convert_mode (str): The pilmode to be used for reading image file into
            numpy array. Default value is 'RGB'.  Find more information at
            https://imageio.readthedocs.io/en/stable/format_png-pil.html
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

    HTTP_METHODS = ["POST"]

    def __init__(
        self,
        input_names=("image",),
        accept_image_formats=None,
        convert_mode="RGB",
        div=True,
        cls=None,
        after_open=None,
    ):
        self.imread = _import_imageio_imread()
        self.fastai_vision = _import_fastai_vision()

        self.input_names = input_names
        self.convert_mode = convert_mode
        self.div = div
        self.cls = cls
        self.accept_image_formats = (
            accept_image_formats or get_default_accept_image_formats()
        )
        self.after_open = after_open

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
        return ['imageio', 'fastai']

    def handle_request(self, request, func):
        input_streams = []
        for filename in self.input_names:
            file = request.files.get(filename)
            if file is not None:
                file_name = secure_filename(file.filename)
                verify_image_format_or_raise(file_name, self.accept_image_formats)
                input_streams.append(BytesIO(file.read()))

        if len(input_streams) == 0:
            data = request.get_data()
            if data:
                input_streams = (data,)
            else:
                raise BadInput(
                    "BentoML#ImageHandler unexpected HTTP request: %s" % request
                )

        input_data = []
        for input_stream in input_streams:
            data = self.imread(input_stream, pilmode=self.convert_mode)

            if self.after_open:
                data = self.after_open(data)

            data = self.fastai_vision.pil2tensor(data, np.float32)

            if self.div:
                data = data.div_(255)

            if self.cls:
                data = self.cls(data)
            else:
                data = self.fastai_vision.Image(data)
            input_data.append(data)

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

        image_array = self.fastai_vision.open_image(
            fn=file_path,
            convert_mode=self.convert_mode,
            div=self.div,
            after_open=self.after_open,
            cls=self.cls or self.fastai_vision.Image,
        )

        result = func(image_array)
        if parsed_args.output == "json":
            result = api_func_result_to_json(result)
        else:
            result = str(result)
        print(result)

    def handle_aws_lambda_event(self, event, func):
        if event["headers"].get("Content-Type", "").startswith("images/"):
            image_data = self.imread(
                base64.decodebytes(event["body"]), pilmode=self.pilmode
            )
        else:
            raise BadInput(
                "BentoML currently doesn't support Content-Type: {content_type} for "
                "AWS Lambda".format(content_type=event["headers"]["Content-Type"])
            )

        if self.after_open:
            image_data = self.after_open(image_data)

        image_data = self.fastai_vision.pil2tensor(image_data, np.float32)
        if self.div:
            image_data = image_data.div_(255)
        if self.cls:
            image_data = self.cls(image_data)
        else:
            image_data = self.fastai_vision.Image(image_data)

        result = func(image_data)
        json_output = api_func_result_to_json(result)
        return {"statusCode": 200, "body": json_output}
