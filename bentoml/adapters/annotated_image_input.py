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


def has_json_extension(file_name: str):
    """
    Check if file's extension is an acceptable JSON extension
    (currently only .json is allowed)
    """
    _, extension = os.path.splitext(file_name)

    if extension.lower() in [".json"]:
        return True

    return False


def has_image_extension(file_name: str, accept_format_list: [str]):
    """
    Check if file's extension is within the provided accept_format_list
    """
    _, extension = os.path.splitext(file_name)
    if extension.lower() in accept_format_list:
        return True

    return False


def read_json_file(json_file):
    """
    Read the provided JSON file and return the parsed Python object
    json_file can be any text or binary file that supports .read()
    """
    try:
        parsed = json.load(json_file)
    except (json.JSONDecodeError, UnicodeDecodeError):
        raise BadInput("BentoML#AnnotatedImageInput received invalid JSON file")

    return parsed


def verify_image_format_or_raise(file_name: str, accept_format_list: [str]):
    """
    Raise error if file's extension is not in the provided accept_format_list
    """
    _, extension = os.path.splitext(file_name)
    if extension.lower() not in accept_format_list:
        raise BadInput(
            "Input file not in supported format list: {}".format(accept_format_list)
        )


class AnnotatedImageInput(BaseInputAdapter):
    """Transform incoming image data from http request, cli or lambda event into a
    numpy array, while allowing an optional JSON file for image annotations (such
    as object bounding boxes, class labels, etc.)

    Transforms input image file into a numpy array, and loads JSON file as
    a JSON serializable Python object, providing them to user-defined
    API functions.

    Args:
        image_input_name (string): An acceptable input name for HTTP request
            and function keyword argument. Default value is "image"
        annotation_input_name (string): An acceptable input name for HTTP request
            and function keyword argument. Default value is "annotations"
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

    Example:

        >>> from bentoml import BentoService, api, artifacts
        >>> from bentoml.artifact import TensorflowArtifact
        >>> from bentoml.adapters import AnnotatedImageInput
        >>>
        >>> CLASS_NAMES = ['cat', 'dog']
        >>>
        >>> @artifacts([TensorflowArtifact('classifer')])
        >>> class PetClassification(BentoService):
        >>>    @api(input=AnnotatedImageInput())
        >>>    def predict(self, image: Numpy.array, annotations: JsonSerializable):
        >>>        cropped_pets = some_pet_finder(image, annotations)
        >>>        results = self.artifacts.classifer.predict(cropped_pets)
        >>>        return [CLASS_NAMES[r] for r in results]
        >>>

        The endpoint could then be used with an HTML form that sends multipart data,
        like the example below

        >>> <form action="http://localhost:8000" method="POST"
        >>>       enctype="multipart/form-data">
        >>>     <input name="image" type="file">
        >>>     <input name="annotations" type="file">
        >>>     <input type="submit">
        >>> </form>

        Or the following cURL command

        >>> curl -F image=@image.png
        >>>      -F annotations=@annotations.json
        >>>      http://localhost:8000/predict
    """

    HTTP_METHODS = ["POST"]
    BATCH_MODE_SUPPORTED = True

    def __init__(
        self,
        accept_image_formats=None,
        image_input_name="image",
        annotation_input_name="annotations",
        pilmode="RGB",
        is_batch_input=False,
        **base_kwargs,
    ):
        assert imageio, "`imageio` dependency can be imported"

        super(AnnotatedImageInput, self).__init__(
            is_batch_input=is_batch_input, **base_kwargs
        )

        self.pilmode = pilmode
        self.image_input_name = image_input_name
        self.annotation_input_name = annotation_input_name
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
                        self.image_input_name: {"type": "string", "format": "binary"},
                        self.annotation_input_name: {
                            "type": "string",
                            "format": "binary",
                        },
                    },
                }
            },
        }

    @property
    def pip_dependencies(self):
        return ["imageio"]

    def _load_image_and_json_data(self, request: Request):
        if len(request.files) == 0:
            raise BadInput(
                "BentoML#AnnotatedImageInput unexpected HTTP request format."
            )
        elif len(request.files) > 2:
            raise BadInput(
                "Too many input files. AnnotatedImageInput takes one image file and an "
                "optional JSON annotation file"
            )

        json_file = None
        image_file = None

        for f in iter(request.files.values()):
            if f and hasattr(f, "mimetype") and isinstance(f.mimetype, str):
                file_name = secure_filename(f.filename)
                if re.match("image/", f.mimetype) or (
                    f.mimetype == "application/octet-stream"
                    and has_image_extension(file_name, self.accept_image_formats)
                ):
                    if image_file:
                        raise BadInput(
                            "BentoML#AnnotatedImageInput received two images instead "
                            "of an image file and JSON file"
                        )
                    image_file = f
                elif f.mimetype == "application/json" or (
                    f.mimetype == "application/octet-stream"
                    and has_json_extension(file_name)
                ):
                    if json_file:
                        raise BadInput(
                            "BentoML#AnnotatedImageInput received two JSON files "
                            "instead of an image file and JSON file"
                        )
                    json_file = f
                else:
                    raise BadInput(
                        "BentoML#AnnotatedImageInput received unexpected file of type "
                        f"{f.mimetype} with filename {f.filename}.\n"
                        "AnnotatedInputAdapter expects an 'image/*' file and optional "
                        "'application/json' file.  Alternatively, it can accept two "
                        "'application/octet-stream' files, one with a valid image "
                        "extension and one with a '.json' extension respectively"
                    )
            else:
                raise BadInput(
                    "BentoML#AnnotatedImageInput unexpected HTTP request format"
                )

        if not image_file:
            raise BadInput("BentoML#AnnotatedImageInput requires an image file")

        image_file_name = secure_filename(image_file.filename)
        verify_image_format_or_raise(image_file_name, self.accept_image_formats)
        input_stream = image_file.stream
        input_image = imageio.imread(input_stream, pilmode=self.pilmode)

        if json_file:
            input_json = read_json_file(json_file)
            return {
                self.image_input_name: input_image,
                self.annotation_input_name: input_json,
            }

        return {self.image_input_name: input_image}

    def handle_batch_request(
        self, requests: Iterable[SimpleRequest], func: callable
    ) -> Iterable[SimpleResponse]:
        """
        Batch version of handle_request
        """
        input_datas = []
        slices = []

        for i, req in enumerate(requests):
            if not req.data:
                slices.append(None)
                input_datas.append(None)
                continue
            request = Request.from_values(
                input_stream=BytesIO(req.data),
                content_length=len(req.data),
                headers=req.headers,
            )
            try:
                input_data = self._load_image_and_json_data(request)
            except BadInput:
                slices.append(None)
                input_datas.append(None)
                continue

            input_datas.append(input_data)
            slices.append(i)

        results = [func(**d) if d else {} for d in input_datas]

        return self.output_adapter.to_batch_response(
            result_conc=results,
            slices=slices,
            fallbacks=[None] * len(slices),
            requests=requests,
        )

    def handle_request(self, request, func):
        """Handle http request that has one image file. It will convert image into
        a ndarray for the function to consume

        Args:
            request: incoming request object.
            func: function that will take ndarray as its arg.
            options: configuration for handling request object.
        Return:
            response object
        """
        input_data = self._load_image_and_json_data(request)
        result = func(**input_data)
        return self.output_adapter.to_response(result, request)

    def handle_cli(self, args, func):
        """Handles an CLI command call, convert CLI arguments into
        corresponding data format that user API function is expecting, and
        prints the API function result to console output

        Processes one image file, or one image file with associated JSON
        annotations

        :param args: CLI arguments
        :param func: user API function
        """
        parser = argparse.ArgumentParser()
        parser.add_argument("--" + self.image_input_name, required=True)
        parser.add_argument("--" + self.annotation_input_name, required=False)
        args, unknown_args = parser.parse_known_args(args)

        image_array = []

        image_path = os.path.expanduser(getattr(args, self.image_input_name))

        verify_image_format_or_raise(image_path, self.accept_image_formats)

        if not os.path.isabs(image_path):
            image_path = os.path.abspath(image_path)

        image_array = imageio.imread(image_path, pilmode=self.pilmode)

        if getattr(args, self.annotation_input_name):
            json_data = {}
            json_path = os.path.expanduser(getattr(args, self.annotation_input_name))
            if not os.path.isabs(json_path):
                json_path = os.path.abspath(json_path)
            with open(json_path, "r") as content_file:
                json_data = read_json_file(content_file)

            result = func(image_array, json_data)
        else:
            result = func(image_array)

        return self.output_adapter.to_cli(result, unknown_args)

    def handle_aws_lambda_event(self, event, func):
        """Handles a Lambda event, convert event dict into corresponding
        data format that user API function is expecting, and use API
        function result as response
        :param event: AWS lambda event data of the python `dict` type with
        an event body including an "image" and optional "json" file
        :param func: user API function
        """
        content_type = event['headers']['Content-Type']

        if "multipart/form-data" in content_type:
            request = Request.from_values(
                data=event['body'], content_type=content_type, headers=event['headers']
            )

            input_data = self._load_image_and_json_data(request)
            result = func(**input_data)

            return self.output_adapter.to_aws_lambda_event(result, event)
        else:
            raise BadInput(
                "AnnotatedImageInput only supports multipart/form-data input, "
                f"received {content_type}"
            )
