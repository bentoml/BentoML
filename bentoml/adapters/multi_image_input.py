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
from typing import Iterable

from requests_toolbelt.multipart import decoder
from werkzeug import Request
from werkzeug.utils import secure_filename

from bentoml.adapters.base_input import BaseInputAdapter
from bentoml.adapters.image_input import (
    get_default_accept_image_formats,
    verify_image_format_or_raise,
    _import_imageio_imread,
)
from bentoml.exceptions import BadInput
from bentoml.marshal.utils import SimpleRequest, SimpleResponse

imread = _import_imageio_imread()


class MultiImageInput(BaseInputAdapter):
    """
    Warning: this is a not yet implemented, follow issue #806 for updates
    Transform incoming images data from http request, cli or lambda event into numpy
    array.

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
        ImportError: imageio package is required to use ImageInput

    Example usage:
        ```
        class MyService(BentoService):
            @bentoml.api(input=MultiImageInput(input_names=('imageX', 'imageY')))
            def predict(self, image_groups):
                for image_group in image_groups:
                    image_array_x = image_group['imageX']
                    image_array_y = image_group['imageY']
        ```
    """

    def __init__(
            self,
            input_names=("image",),
            accepted_image_formats=None,
            pilmode="RGB",
            is_batch_input=False,
            **base_kwargs,
    ):
        if is_batch_input:
            raise ValueError('ImageInput can not accept batch inputs')
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

    def read_file(self, name: str, file):
        safe_name = secure_filename(name)
        verify_image_format_or_raise(safe_name, self.accepted_image_formats)
        return imread(file, pilmode=self.pilmode)

    def handle_batch_request(
            self, requests: Iterable[SimpleRequest], func
    ) -> Iterable[SimpleResponse]:
        raise NotImplementedError(
            "The batch processing architecture does not currently support multipart "
            "requests, which are required for multi-image requests"
        )

    def handle_cli(self, args, func):
        parser = argparse.ArgumentParser()
        for input_name in self.input_names:
            parser.add_argument(input_name, required=True)
        args, unknown_args = parser.parse_known_args(args)
        files = {
            input_name: self.read_file(
                pathlib.Path(args[input_name]).name, args[input_name]
            )
            for input_name in self.input_names
        }
        result = func((files,))[0]
        return self.output_adapter.to_cli(result, unknown_args)

    def handle_aws_lambda_event(self, event, func):
        content_type = event.headers['content-type']
        if "multipart/form-data" in content_type:
            files = {}

            request = decoder.MultipartDecoder(event.body, content_type)
            for part in request.parts:
                part: decoder.BodyPart = part
                if part.headers['name'] in self.input_names:
                    files[part.headers['name']] = self.read_file(
                        part.headers['filename'],
                        part.content
                    )
            result = func((files,))[0]
            return self.output_adapter.to_aws_lambda_event(result, event)
        else:
            raise BadInput(
                "Multi-image requests don't support the {} content type".format(content_type)
            )
