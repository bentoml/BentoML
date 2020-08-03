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


class ImageInput(BaseInputAdapter):
    """Transform incoming image data from http request, cli or lambda event into numpy
    array.

    Handle incoming image data from different sources, transform them into numpy array
    and pass down to user defined API functions

    * If you want to operate raw image file stream or PIL.Image objects, use lowlevel
    alternative FileInput.

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
        ImportError: imageio package is required to use ImageInput

    Example:

        ```python
        from bentoml import BentoService, api, artifacts
        from bentoml.artifact import TensorflowArtifact
        from bentoml.adapters import ImageInput

        CLASS_NAEMS = ['cat', 'dog']

        @artifacts([TensorflowArtifact('classifer')])
        class PetClassification(BentoService):
            @api(input=ImageInput())
            def predict(self, image_ndarrays):
                results = self.artifacts.classifer.predict(image_ndarrays)
                return [CLASS_NAEMS[r] for r in results]
        ```
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
            raise ValueError('ImageInput can not accpept batch inputs')
        super(ImageInput, self).__init__(is_batch_input=is_batch_input, **base_kwargs)
        if 'input_names' in base_kwargs:
            raise TypeError(
                "ImageInput doesn't take input_names as parameters since bentoml 0.8."
                "Update your Service definition "
                "or use LegacyImageInput instead(not recommended)."
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
            "image/*": {"schema": {"type": "string", "format": "binary"}},
            "multipart/form-data": {
                "schema": {
                    "type": "object",
                    "properties": {
                        "image_file": {"type": "string", "format": "binary"}
                    },
                }
            },
        }

    @property
    def pip_dependencies(self):
        return ["imageio"]

    def _load_image_data(self, request: Request):
        if len(request.files):
            if len(request.files) != 1:
                raise BadInput(
                    "ImageInput requires one and at least one image file at a time, "
                    "if you just upgraded from bentoml 0.7, you may need to use "
                    "FileInput or LegacyImageInput instead"
                )
            input_file = next(iter(request.files.values()))
            if not input_file:
                raise BadInput("BentoML#ImageInput unexpected HTTP request format")
            file_name = secure_filename(input_file.filename)
            verify_image_format_or_raise(file_name, self.accept_image_formats)
            input_stream = input_file.stream
        else:
            data = request.get_data()
            if not data:
                raise BadInput("BentoML#ImageInput unexpected HTTP request format")
            else:
                input_stream = data

        input_data = imageio.imread(input_stream, pilmode=self.pilmode)
        return input_data

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
                input_data = self._load_image_data(request)
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
        input_data = self._load_image_data(request)
        result = func((input_data,))[0]
        return self.output_adapter.to_response(result, request)

    def handle_cli(self, args, func):
        parser = argparse.ArgumentParser()
        parser.add_argument("--input", required=True, nargs='+')
        parser.add_argument("--batch-size", default=None, type=int)
        parsed_args, unknown_args = parser.parse_known_args(args)
        file_paths = parsed_args.input

        batch_size = (
            parsed_args.batch_size if parsed_args.batch_size else len(file_paths)
        )

        for i in range(0, len(file_paths), batch_size):
            step_file_paths = file_paths[i : i + batch_size]
            image_arrays = []
            for file_path in step_file_paths:
                verify_image_format_or_raise(file_path, self.accept_image_formats)
                if not os.path.isabs(file_path):
                    file_path = os.path.abspath(file_path)

                image_arrays.append(imageio.imread(file_path, pilmode=self.pilmode))

            results = func(image_arrays)
            for result in results:
                return self.output_adapter.to_cli(result, unknown_args)

    def handle_aws_lambda_event(self, event, func):
        if event["headers"].get("Content-Type", "").startswith("images/"):
            image = imageio.imread(
                base64.decodebytes(event["body"]), pilmode=self.pilmode
            )
        else:
            raise BadInput(
                "BentoML currently doesn't support Content-Type: {content_type} for "
                "AWS Lambda".format(content_type=event["headers"]["Content-Type"])
            )

        result = func((image,))[0]
        return self.output_adapter.to_aws_lambda_event(result, event)
