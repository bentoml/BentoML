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

import pathlib
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

    Parameters
    ----------
    input_names : List[str]
        list of input names. For HTTP they are form input names. For CLI
        they are CLI args --input-<name1> or --input-file-<name1>

    allow_none : bool
        accept HTTP requests or AWS Lambda events without all files
        provided. Does not take effect on CLI.

    Examples
    ----------
    Service using MultiFileInput:

    .. code-block:: python

        from typing import List

        from PIL import Image
        import numpy as np
        import bentoml
        from bentoml.types import FileLike
        from bentoml.framework.pytroch import PytorchModelArtifact
        from bentoml.adapters import MultiFileInput

        @bentoml.env(pip_packages=['torch', 'pillow', 'numpy'])
        @bentoml.artifacts([PytorchModelArtifact('classifier')])
        class PyTorchFashionClassifier(bentoml.BentoService):
            @bentoml.api(
                input=MultiFileInput(input_names=['image', 'json']), batch=True)
            def predict(self, image_list: List[FileLike], json_list: List[FileLike]):
                inputs = []
                for img_io, json_io in zip(image_list, json_list):
                    img = Image.open(img_io)
                    json_obj = json.load(json_io)
                    inputs.append([img, json_obj])
                outputs = self.artifacts.classifier(inputs)
                return outputs

    Query with HTTP request performed by cURL::

        curl -i \\
          -F image=@test.jpg \\
          -F json=@test.json \\
          localhost:5000/predict

    OR by an HTML form that sends multipart data:

    .. code-block:: html

        <form action="http://localhost:8000" method="POST"
              enctype="multipart/form-data">
            <input name="image" type="file">
            <input name="json" type="file">
            <input type="submit">
        </form>

    OR by python requests:

    .. code-block:: python

        import requests

        with open("test.jpg", "rb") as f:
            image_bytes = f.read()
        with open("anno.json", "rb") as f:
            json_bytes = f.read()

        files = {
            "image": ("test.jpg", image_bytes),
            "json": ("test.json", json_bytes),
        }
        response = requests.post(your_url, files=files)

    Query with CLI command::

        bentoml run PyTorchFashionClassifier:latest predict \\
          --input-file-image test.jpg \\
          --input-file-json test.json

    OR infer all file pairs under a folder with ten pairs each batch::

        bentoml run PyTorchFashionClassifier:latest predict --max-batch-size 10 \\
          --input-file-image folder/*.jpg \\
          --input-file-json folder/*.json

    Note: jpg files and json files should be in same prefix like this::

        folder:
            - apple.jpg
            - apple.json
            - banana.jpg
            - banana.json
            ...

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
        input_, input_file = parse_cli_inputs(cli_args, self.input_names)
        for t in self.from_inference_job(input_=input_, input_file=input_file):
            t.cli_args = cli_args
            yield t

    def from_inference_job(  # pylint: disable=arguments-differ
        self, input_=None, input_file=None, **extra_args
    ) -> Iterator[InferenceTask[FileLike]]:
        if input_file is not None:
            for ds in zip(*input_file):
                uris = (pathlib.Path(d).absolute().as_uri() for d in ds)
                fs = tuple(FileLike(uri=uri) for uri in uris)
                yield InferenceTask(data=fs, inference_job_args=extra_args)
        else:
            for ds in zip(*input_):
                fs = tuple(FileLike(bytes_=d.encode()) for d in ds)
                yield InferenceTask(data=fs, inference_job_args=extra_args)

    def extract_user_func_args(self, tasks: Sequence[MultiFileTask]) -> ApiFuncArgs:
        args = tuple(map(tuple, zip(*map(lambda t: t.data, tasks))))
        if not args:
            args = (tuple(),) * len(self.input_names)
        return args
