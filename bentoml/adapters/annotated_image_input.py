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

import re
import json
import os
import argparse
import base64
from io import BytesIO
from typing import Iterable

from werkzeug.utils import secure_filename
from werkzeug.wrappers import Request

from bentoml import config
from bentoml.utils.lazy_loader import LazyLoader
from bentoml.marshal.utils import SimpleRequest, SimpleResponse
from bentoml.exceptions import BadInput
from bentoml.adapters.base_input import BaseInputAdapter

# BentoML optional dependencies, using lazy load to avoid ImportError
imageio = LazyLoader('imageio', globals(), 'imageio')


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
        for extension in config("apiserver")
        .get("default_image_input_accept_file_extensions")
        .split(",")
    ]


class AnnotatedImageInput(BaseInputAdapter):
    """Transform incoming image data from http request, cli or lambda event into numpy
    array.

    Handle incoming image data from different sources, transform them into numpy array
    and pass down to user defined API functions

    Args:
        accept_image_formats (string[]):  A list of acceptable image formats.
            Default value is loaded from bentoml config
            'apiserver/default_image_input_accept_file_extensions', which is
            set to ['.jpg', '.png', '.jpeg', '.tiff', '.webp', '.bmp'] by default.
            List of all supported format can be found here:
            https://imageio.readthedocs.io/en/stable/formats.html
        pilmode (string): The pilmode to be used for reading image file into numpy
            array. Default value is 'RGB'.  Find more information at:
            https://imageio.readthedocs.io/en/stable/format_png-pil.html

    Raises:
        ImportError: imageio package is required to use AnnotatedImageInput
    """

    HTTP_METHODS = ["POST"]
    BATCH_MODE_SUPPORTED = True

    def __init__(
        self,
        accept_image_formats=None,
        pilmode="RGB",
        is_batch_input=False,
        **base_kwargs,
    ):
        assert imageio, "`imageio` dependency can be imported"

        if is_batch_input:
            raise ValueError('AnnotatedImageInput can not accpept batch inputs')
        super(AnnotatedImageInput, self).__init__(is_batch_input=is_batch_input, **base_kwargs)
        if 'input_names' in base_kwargs:
            raise TypeError(
                "AnnotatedImageInput doesn't take input_names as parameters since bentoml 0.8."
                "Update your Service definition "
                "or use LegacyAnnotatedImageInput instead(not recommended)."
            )

        self.pilmode = pilmode
        self.accept_image_formats = (
            accept_image_formats or get_default_accept_image_formats()
        )

    @property
    def config(self):
        return {
            # Converting to list, google.protobuf.Struct does not work with tuple type
            "accept_image_formats": self.accept_image_formats,
            "pilmode": self.pilmode,
        }

    @property
    def request_schema(self):
        return {
            "multipart/form-data": {
                "schema": {
                    "type": "object",
                    "properties": {
                        "image_file": {"type": "string", "format": "binary"},
                        "json_file": {"type": "string", "format": "binary"}
                    },
                }
            },
        }

    @property
    def pip_dependencies(self):
        return ["imageio"]

    def _load_image_and_json_data(self, request: Request):
        if len(request.files) == 0:
            raise BadInput("BentoML#AnnotatedImageInput unexpected HTTP request format.")
        elif len(request.files) > 2:
            raise BadInput(
                "Too many input files. AnnotatedImageInput takes one image file and an \
                optional JSON annotation file"
            )

        json_file = None
        image_file = None

        for f in iter(request.files.values()):
            if re.match("image/", f.mimetype):
                if image_file:
                    raise BadInput("BentoML#AnnotatedImageInput received two images instead of an image file and JSON file")
                image_file = f
            elif f.mimetype == "application/json":
                if json_file:
                    raise BadInput("BentoML#AnnotatedImageInput received two JSON files instead of an image file and JSON file")
                json_file = f
            else:
                raise BadInput("BentoML#AnnotatedImageInput received unexpected file type.")

        if not image_file:
            raise BadInput("BentoML#AnnotatedImageInput requires an image file")

        file_name = secure_filename(image_file.filename)
        verify_image_format_or_raise(file_name, self.accept_image_formats)
        input_stream = image_file.stream
        input_image = imageio.imread(input_stream, pilmode=self.pilmode)
        input_json = {}

        if json_file:
            try:
                input_json = json.load(json_file)
            except (json.JSONDecodeError, UnicodeDecodeError):
                raise BadInput("BentoML#AnnotatedImageInput received invalid JSON file")

        return (input_image, input_json)

    def handle_batch_request(
        self, requests: Iterable[SimpleRequest], func: callable
    ) -> Iterable[SimpleResponse]:
        """
        Batch version of handle_request
        """
        input_datas = []
        ids = []
        for i, req in enumerate(requests):
            if not req.data:
                ids.append(None)
                continue
            request = Request.from_values(
                input_stream=BytesIO(req.data),
                content_length=len(req.data),
                headers=req.headers,
            )
            try:
                input_data = self._load_image_and_json_data(request)
            except BadInput:
                ids.append(None)
                continue

            input_datas.append(input_data)
            ids.append(i)

        results = func(input_datas) if input_datas else []
        return self.output_adapter.to_batch_response(results, ids, requests)

    def handle_request(self, request, func):
        """Handle http request that has one image file. It will convert image into a
        ndarray for the function to consume.

        Args:
            request: incoming request object.
            func: function that will take ndarray as its arg.
            options: configuration for handling request object.
        Return:
            response object
        """
        input_data = self._load_image_and_json_data(request)
        result = func(*input_data)[0]
        return self.output_adapter.to_response(result, request)

    def handle_cli(self, args, func):
        """Handles an CLI command call, convert CLI arguments into
        corresponding data format that user API function is expecting, and
        prints the API function result to console output
        :param args: CLI arguments
        :param func: user API function
        """
        parser = argparse.ArgumentParser()
        parser.add_argument("--image", required=True)
        parser.add_argument("--json", required=False)
        args, unknown_args = parser.parse_known_args(args)

        image_array = []
        json_data = {}

        image_path = os.path.expanduser(args.image)

        verify_image_format_or_raise(image_path, self.accept_image_formats)

        if not os.path.isabs(image_path):
            image_path = os.path.abspath(image_path)

        image_array = imageio.imread(image_path, pilmode=self.pilmode)


        if args.json:
            json_path = os.path.expanduser(args.json)
            if not os.path.isabs(json_path):
                json_path = os.path.abspath(json_path)
            try:
                with open(json_path, "r") as content_file:
                    json_data = json.load(content_file)
            except (json.JSONDecodeError, UnicodeDecodeError):
                raise BadInput("BentoML#AnnotatedImageInput received invalid JSON file")

        result = func(image_array,json_data)
        return self.output_adapter.to_cli(result, unknown_args)

    def handle_aws_lambda_event(self, event, func):
        """Handles a Lambda event, convert event dict into corresponding
        data format that user API function is expecting, and use API
        function result as response
        :param event: AWS lambda event data of the python `dict` type
        :param func: user API function
        """
        content_type = event['headers']['Content-Type']
        if "multipart/form-data" in content_type:
            files = {}

            request = Request.from_values(
                data=event['body'], content_type=content_type, headers=event['headers']
            )

            input_data = self._load_image_and_json_data(request)
            result = func(*input_data)[0]

            return self.output_adapter.to_aws_lambda_event(result, event)
        else:
            raise BadInput(
                "Annotated image requests don't support the {} content type".format(
                    content_type
                )
            )

