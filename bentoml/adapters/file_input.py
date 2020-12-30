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
import pathlib
from typing import Iterable, Iterator, Sequence, Tuple

from bentoml.adapters.base_input import BaseInputAdapter
from bentoml.adapters.utils import decompress_gzip_request
from bentoml.types import AwsLambdaEvent, FileLike, HTTPRequest, InferenceTask

ApiFuncArgs = Tuple[
    Sequence[FileLike],
]


class FileInput(BaseInputAdapter):
    """Convert incoming file data from http request, cli or lambda event into file
    stream object and pass down to user defined API functions

    Parameters
    ----------
    None

    Examples
    ----------
    Service using FileInput:

    .. code-block:: python

        import bentoml
        from PIL import Image
        import numpy as np

        from bentoml.frameworks.pytorch import PytorchModelArtifact
        from bentoml.adapters import FileInput

        FASHION_MNIST_CLASSES = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
                                 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

        @bentoml.env(pip_packages=['torch', 'pillow', 'numpy'])
        @bentoml.artifacts([PytorchModelArtifact('classifier')])
        class PyTorchFashionClassifier(bentoml.BentoService):

            @bentoml.api(input=FileInput(), batch=True)
            def predict(self, file_streams):
                img_arrays = []
                for fs in file_streams:
                    im = Image.open(fs).convert(mode="L").resize((28, 28))
                    img_array = np.array(im)
                    img_arrays.append(img_array)

                inputs = np.stack(img_arrays, axis=0)

                outputs = self.artifacts.classifier(inputs)
                return [FASHION_MNIST_CLASSES[c] for c in outputs]

    OR use FileInput with ``batch=False`` (the default):

    .. code-block:: python

        @bentoml.api(input=FileInput(), batch=False)
        def predict(self, file_stream):
            im = Image.open(file_stream).convert(mode="L").resize((28, 28))
            img_array = np.array(im)
            inputs = np.stack([img_array], axis=0)
            outputs = self.artifacts.classifier(inputs)
            return FASHION_MNIST_CLASSES[outputs[0]]

    Query with HTTP request performed by cURL::

        curl -i \\
          --header "Content-Type: image/jpeg" \\
          --request POST \\
          --data-binary @test.jpg \\
          localhost:5000/predict

    OR::

        curl -i \\
          -F image=@test.jpg \\
          localhost:5000/predict

    OR by an HTML form that sends multipart data:

    .. code-block:: html

        <form action="http://localhost:8000" method="POST"
              enctype="multipart/form-data">
            <input name="image" type="file">
            <input type="submit">
        </form>

    OR by python requests:

    .. code-block:: python

        import requests

        with open("test.jpg", "rb") as f:
            image_bytes = f.read()

        files = {
            "image": ("test.jpg", image_bytes),
        }
        response = requests.post(your_url, files=files)

    Query with CLI command::

        bentoml run PyTorchFashionClassifier:latest predict --input-file test.jpg

    OR infer all images under a folder with ten images each batch::

        bentoml run PyTorchFashionClassifier:latest predict \\
          --input-file folder/*.jpg --max-batch-size 10

    """

    HTTP_METHODS = ["POST"]
    BATCH_MODE_SUPPORTED = True

    @property
    def request_schema(self):
        return {
            "multipart/form-data": {
                "schema": {
                    "type": "object",
                    "properties": {"file": {"type": "string", "format": "binary"}},
                }
            },
            "*/*": {"schema": {"type": "string", "format": "binary"}},
        }

    @decompress_gzip_request
    def from_http_request(self, req: HTTPRequest) -> InferenceTask[FileLike]:
        if req.headers.content_type == 'multipart/form-data':
            _, _, files = HTTPRequest.parse_form_data(req)
            if len(files) != 1:
                task = InferenceTask(data=None)
                task.discard(
                    http_status=400,
                    err_msg=f"BentoML#{self.__class__.__name__} requires one and at"
                    " least one file at a time, if you just upgraded from"
                    " bentoml 0.7, you may need to use MultiFileAdapter instead",
                )
            else:
                input_file = next(iter(files.values()))
                task = InferenceTask(http_headers=req.headers, data=input_file)
        elif req.body:
            task = InferenceTask(
                http_headers=req.headers, data=FileLike(bytes_=req.body),
            )
        else:
            task = InferenceTask(data=None)
            task.discard(
                http_status=400,
                err_msg=f'BentoML#{self.__class__.__name__} unexpected HTTP request'
                ' format',
            )
        return task

    def from_aws_lambda_event(self, event: AwsLambdaEvent) -> InferenceTask[FileLike]:
        f = FileLike(bytes_=base64.decodebytes(event.get('body', "")))
        return InferenceTask(aws_lambda_event=event, data=f)

    def from_cli(self, cli_args: Tuple[str]) -> Iterator[InferenceTask[FileLike]]:
        import argparse

        parser = argparse.ArgumentParser()
        input_g = parser.add_mutually_exclusive_group(required=True)
        input_g.add_argument('--input', nargs="+", type=str)
        input_g.add_argument('--input-file', nargs="+")

        parsed_args, _ = parser.parse_known_args(list(cli_args))

        for t in self.from_inference_job(
            input_=parsed_args.input, input_file=parsed_args.input_file,
        ):
            t.cli_args = cli_args
            yield t

    def from_inference_job(  # pylint: disable=arguments-differ
        self, input_=None, input_file=None, **extra_args
    ) -> Iterator[InferenceTask[str]]:
        '''
        Generate InferenceTask from calling bentom_svc.run(input_=None, input_file=None)

        Parameters
        ----------
        input_ : str
            The input value

        input_file : str
            The URI/path of the input file

        extra_args : dict
            Additional parameters

        '''
        if input_file is not None:
            for d in input_file:
                uri = pathlib.Path(d).absolute().as_uri()
                yield InferenceTask(
                    inference_job_args=extra_args, data=FileLike(uri=uri)
                )
        else:
            for d in input_:
                yield InferenceTask(
                    inference_job_args=extra_args, data=FileLike(bytes_=d.encode()),
                )

    def extract_user_func_args(
        self, tasks: Iterable[InferenceTask[FileLike]]
    ) -> ApiFuncArgs:
        return (tuple(t.data for t in tasks),)
