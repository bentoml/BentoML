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

from typing import Iterator, Sequence, Tuple

from bentoml.adapters.base_input import BaseInputAdapter, parse_cli_inputs
from bentoml.adapters.utils import decompress_gzip_request
from bentoml.types import AwsLambdaEvent, FileLike, HTTPRequest, InferenceTask

ApiFuncArgs = Tuple[Sequence[FileLike], ...]
MultiFileTask = InferenceTask[Tuple[FileLike, ...]]


class MultiFileInput(BaseInputAdapter):
    """ Low level input adapters that transform incoming files data from http request,
    CLI or AWS lambda event into binary stream objects, then pass down to user defined
    API functions.

    Args:
        input_names: list of input names. For HTTP they are form input names. For CLI
            they are CLI args --input-<name1> or --input-file-<name1>
        allow_none: accept HTTP requests or AWS Lambda events without all files
            provided. Does not take effect on CLI.

    Example:

    >>> import bentoml
    >>> from PIL import Image
    >>> import numpy as np
    >>>
    >>> from bentoml.framework.pytroch import PytorchModelArtifact
    >>> from bentoml.adapters import MultiFileInput
    >>>
    >>> @bentoml.env(pip_packages=['torch', 'pillow', 'numpy'])
    >>> @bentoml.artifacts([PytorchModelArtifact('classifier')])
    >>> class PyTorchFashionClassifier(bentoml.BentoService):
    >>>     @bentoml.api(
    >>>         input=MultiFileInput(input_names=['image', 'json']), batch=True)
    >>>     def predict(self, image_list, json_list):
    >>>         for img_io, json_io in zip(image_list, json_list):
    >>>             img = Image.open(img_io)
    >>>             json_obj = json.load(json_io)

    """

    HTTP_METHODS = ["POST"]
    BATCH_MODE_SUPPORTED = True

    def __init__(
        self, input_names: Sequence[str], allow_none: bool = False, **base_kwargs,
    ):

        super().__init__(**base_kwargs)
        self.input_names = input_names
        self.allow_none = allow_none

    @property
    def config(self):
        return {
            # Converting to list, google.protobuf.Struct does not work with tuple type
            "input_names": list(self.input_names)
        }

    @property
    def request_schema(self):
        return {
            "multipart/form-data": {
                "schema": {
                    "type": "object",
                    "properties": {
                        k: {"type": "string", "format": "binary"}
                        for k in self.input_names
                    },
                }
            },
        }

    @decompress_gzip_request
    def from_http_request(self, req: HTTPRequest) -> MultiFileTask:
        if req.headers.content_type != 'multipart/form-data':
            task = InferenceTask(data=None)
            task.discard(
                http_status=400,
                err_msg=f"BentoML#{self.__class__.__name__} only accepts requests "
                "with Content-Type: multipart/form-data",
            )
        else:
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
                task = InferenceTask(http_headers=req.headers, data=files,)
        return task

    def from_aws_lambda_event(self, event: AwsLambdaEvent) -> MultiFileTask:
        request = HTTPRequest(
            headers=tuple((k, v) for k, v in event.get('headers', {}).items()),
            body=event['body'],
        )
        return self.from_http_request(request)

    def from_cli(self, cli_args: Sequence[str]) -> Iterator[MultiFileTask]:
        for inputs in parse_cli_inputs(cli_args, self.input_names):
            yield InferenceTask(cli_args=cli_args, data=inputs)

    def extract_user_func_args(self, tasks: Sequence[MultiFileTask]) -> ApiFuncArgs:
        args = tuple(map(tuple, zip(*map(lambda t: t.data, tasks))))
        if not args:
            args = (tuple(),) * len(self.input_names)
        return args
