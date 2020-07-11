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
import argparse
import pathlib
from io import BytesIO
from typing import Iterable

from werkzeug import Request
from werkzeug.utils import secure_filename

from bentoml.utils.lazy_loader import LazyLoader
from bentoml.adapters.base_input import BaseInputAdapter
from bentoml.adapters.image_input import (
    get_default_accept_image_formats,
    verify_image_format_or_raise,
)
from bentoml.exceptions import BadInput
from bentoml.marshal.utils import SimpleRequest, SimpleResponse


# BentoML optional dependencies, using lazy load to avoid ImportError
imageio = LazyLoader('imageio', globals(), 'imageio')


class MultiImageInput(BaseInputAdapter):
    """
    Args:
        input_names (string[]): A tuple of acceptable input name for HTTP request.
            Default value is (image,)
        accepted_image_formats (string[]):  A list of acceptable image formats.
            Default value is loaded from bentoml config
            'apiserver/default_image_input_accept_file_extensions', which is
            set to ['.jpg', '.png', '.jpeg', '.tiff', '.webp', '.bmp'] by default.
            List of all supported format can be found here:
            https://imageio.readthedocs.io/en/stable/formats.html
        pilmode (string): The pilmode to be used for reading image file into numpy
            array. Default value is 'RGB'.  Find more information at:
            https://imageio.readthedocs.io/en/stable/format_png-pil.html

    Raises:
        ImportError: imageio package is required to use MultiImageInput

    Example usage:

    >>> from bentoml import BentoService
    >>> import bentoml
    >>>
    >>> class MyService(BentoService):
    >>>     @bentoml.api(input=MultiImageInput(input_names=('imageX', 'imageY')))
    >>>     def predict(self, image_groups):
    >>>         for image_group in image_groups:
    >>>             image_array_x = image_group['imageX']
    >>>             image_array_y = image_group['imageY']

    The endpoint could then be used with an HTML form that sends multipart data, like
    the example below


    >>> <form action="http://localhost:8000" method="POST"
    >>>       enctype="multipart/form-data">
    >>>     <input name="imageX" type="file">
    >>>     <input name="imageY" type="file">
    >>>     <input type="submit">
    >>> </form>

    Or the following cURL command

    >>> curl -F imageX=@image_file_x.png
    >>>      -F imageY=@image_file_y.jpg
    >>>      http://localhost:8000
    """

    def __init__(
        self,
        input_names=("image",),
        accepted_image_formats=None,
        pilmode="RGB",
        is_batch_input=False,
        **base_kwargs,
    ):
        super(MultiImageInput, self).__init__(
            is_batch_input=is_batch_input, **base_kwargs
        )
        self.input_names = input_names
        self.pilmode = pilmode
        self.accepted_image_formats = (
            accepted_image_formats or get_default_accept_image_formats()
        )

    def handle_request(self, request: Request, func):
        files = {
            name: self.read_file(file.filename, file.stream)
            for (name, file) in request.files.items()
        }
        result = func((files,))[0]
        return self.output_adapter.to_response(result, request)

    def read_file(self, name: str, file: BytesIO):
        safe_name = secure_filename(name)
        verify_image_format_or_raise(safe_name, self.accepted_image_formats)
        return imageio.imread(file, pilmode=self.pilmode)

    def handle_batch_request(
        self, requests: Iterable[SimpleRequest], func
    ) -> Iterable[SimpleResponse]:
        inputs = []
        slices = []
        for i, req in enumerate(requests):
            content_type = next(
                header[1] for header in req.headers if header[0] == b"Content-Type"
            )

            if b"multipart/form-data" not in content_type:
                slices.append(None)
            else:
                files = {}
                request = Request.from_values(
                    data=req.data, content_type=content_type, headers=req.headers,
                )
                for name in request.files:
                    file = request.files[name]
                    files[name] = self.read_file(file.filename, file.stream)
                inputs.append(files)
                slices.append(i)
        results = func(inputs) if inputs else []
        return self.output_adapter.to_batch_response(results, slices, requests)

    def handle_cli(self, args, func):
        """Handles an CLI command call, convert CLI arguments into
        corresponding data format that user API function is expecting, and
        prints the API function result to console output
        :param args: CLI arguments
        :param func: user API function
        """
        parser = argparse.ArgumentParser()
        for input_name in self.input_names:
            parser.add_argument('--' + input_name, required=True)
        args, unknown_args = parser.parse_known_args(args)
        args = vars(args)
        files = {
            input_name: self.read_file(
                pathlib.Path(args[input_name]).name, args[input_name]
            )
            for input_name in self.input_names
        }
        result = func((files,))[0]
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
            for name in request.files:
                file = request.files[name]
                files[name] = self.read_file(file.filename, file.stream)
            result = func((files,))[0]
            return self.output_adapter.to_aws_lambda_event(result, event)
        else:
            raise BadInput(
                "Multi-image requests don't support the {} content type".format(
                    content_type
                )
            )
