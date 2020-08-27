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
import base64
import io
from typing import Tuple, BinaryIO, Sequence, Iterable, List

from bentoml.types import InferenceTask, HTTPRequest, InferenceContext
from bentoml.utils.lazy_loader import LazyLoader
from bentoml.adapters.multi_image_input import MultiImageInput

# BentoML optional dependencies, using lazy load to avoid ImportError
imageio = LazyLoader('imageio', globals(), 'imageio')
numpy = LazyLoader('numpy', globals(), 'numpy')


MultiImgTask = InferenceTask[Tuple[BinaryIO, ...]]  # image file bytes, json bytes
ApiFuncArgs = Tuple[Sequence['numpy.ndarray'], ...]


class LegacyImageInput(MultiImageInput):
    """
    *** This LegacyImageInput is identical to the ImageHandler prior to
    BentoML version 0.8.0, it was kept here to make it easier for users to upgrade.
    If you are starting a new model serving project, use the ImageInput instead.
    LegacyImageInput will be deprecated in release 1.0.0. ***

    Transform incoming image data from http request, cli or lambda event into numpy
    array.

    Handle incoming image data from different sources, transform them into numpy array
    and pass down to user defined API functions

    Args:
        input_names (string[]]): A tuple of acceptable input name for HTTP request.
            Default value is (image,)
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
        ImportError: imageio package is required to use LegacyImageInput

    Example usage:

    >>> from bentoml import BentoService
    >>> import bentoml
    >>>
    >>> class MyService(BentoService):
    >>>     @bentoml.api(input=LegacyImageInput(input_names=('imageX', 'imageY')))
    >>>     def predict(self, imageX, imageY):
    >>>         pass
    >>>

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

    BATCH_MODE_SUPPORTED = False

    def from_http_request(self, reqs: Iterable[HTTPRequest]) -> List[MultiImgTask]:
        tasks = []
        for req in reqs:
            if len(self.input_names) == 1:
                # broad parsing while single input
                if req.parsed_headers.content_type == 'multipart/form-data':
                    _, _, files = HTTPRequest.parse_form_data(req)
                    if not any(files):
                        task = InferenceTask(data=None)
                        task.discard(
                            http_status=400,
                            err_msg=f"BentoML#{self.__class__.__name__} requires inputs"
                            f"fields {self.input_names}",
                        )
                    else:
                        f = next(iter(files.values()))
                        task = InferenceTask(
                            context=InferenceContext(http_headers=req.parsed_headers),
                            data=(f,),
                        )
                else:
                    # for images/*
                    task = InferenceTask(
                        context=InferenceContext(http_headers=req.parsed_headers),
                        data=(io.BytesIO(req.body),),
                    )
            elif req.parsed_headers.content_type == 'multipart/form-data':
                _, _, files = HTTPRequest.parse_form_data(req)
                files = tuple(files.get(k) for k in self.input_names)
                if not any(files):
                    task = InferenceTask(data=None)
                    task.discard(
                        http_status=400,
                        err_msg=f"BentoML#{self.__class__.__name__} requires inputs "
                        f"fields {self.input_names}",
                    )
                elif not all(files) and not self.allow_none:
                    task = InferenceTask(data=None)
                    task.discard(
                        http_status=400,
                        err_msg=f"BentoML#{self.__class__.__name__} requires inputs "
                        f"fields {self.input_names}",
                    )
                else:
                    task = InferenceTask(
                        context=InferenceContext(http_headers=req.parsed_headers),
                        data=files,
                    )
            else:
                task = InferenceTask(data=None)
                task.discard(
                    http_status=400,
                    err_msg=f"BentoML#{self.__class__.__name__} with multiple inputs "
                    "accepts requests with Content-Type: multipart/form-data only",
                )
            tasks.append(task)

        return tasks

    def from_aws_lambda_event(self, events):
        tasks = []
        for event in events:
            if event["headers"].get("Content-Type", "").startswith("images/"):
                img_bytes = base64.b64decode(event["body"])
                img_io = io.BytesIO(img_bytes)
                _, ext = event["headers"]["Content-Type"].split('/')
                img_io.name = f"img.{ext}"
                task = InferenceTask(data=(img_io,))
            else:
                task = InferenceTask(data=None)
                task.discard(
                    http_status=400,
                    err_msg="BentoML currently doesn't support Content-Type: "
                    "{content_type} for AWS Lambda".format(
                        content_type=event["headers"]["Content-Type"]
                    ),
                )
            tasks.append(task)
        return tuple(tasks)
