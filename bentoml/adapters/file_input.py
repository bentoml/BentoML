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

import io
import base64
from io import BytesIO
from typing import Iterable, BinaryIO, Tuple, Iterator, Sequence, List


from bentoml.types import (
    HTTPRequest,
    AwsLambdaEvent,
    InferenceTask,
    InferenceContext,
)
from bentoml.adapters.base_input import BaseInputAdapter, parse_cli_inputs


ApiFuncArgs = Tuple[
    Sequence[BytesIO],
]


class FileInput(BaseInputAdapter[ApiFuncArgs]):
    """Transform incoming file data from http request, cli or lambda event into file
    stream object.

    Handle incoming file data from different sources, transform them into file streams
    and pass down to user defined API functions

    Args:
        None

    Example:

        ```python
        import bentoml
        from PIL import Image
        import numpy as np

        from bentoml.artifact import PytorchModelArtifact
        from bentoml.adapters import FileInput


        FASHION_MNIST_CLASSES = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
                                 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']


        @bentoml.env(pip_dependencies=['torch', 'pillow', 'numpy'])
        @bentoml.artifacts([PytorchModelArtifact('classifier')])
        class PyTorchFashionClassifier(bentoml.BentoService):

            @bentoml.api(input=FileInput())
            def predict(self, file_streams):
                img_arrays = []
                for fs in file_streams:
                    im = Image.open(fs).convert(mode="L").resize((28, 28))
                    img_array = np.array(im)
                    img_arrays.append(img_array)

                inputs = np.stack(img_arrays, axis=0)

                outputs = self.artifacts.classifier(inputs)
                return [FASHION_MNIST_CLASSES[c] for c in outputs]
        ```

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

    def from_http_request(
        self, reqs: Iterable[HTTPRequest]
    ) -> List[InferenceTask[BinaryIO]]:
        tasks = [None] * len(reqs)
        for i, req in enumerate(reqs):
            if req.content_type == 'multipart/form-data':
                _, _, files = HTTPRequest.parse_form_data(req)
                if len(files) != 1:
                    task = InferenceTask(
                        context=InferenceContext(http_headers=req.parsed_headers),
                        data=None,
                    )
                    task.discard(
                        http_status=400,
                        err_msg="ImageHandler requires one and at least one file at a "
                        "time, if you just upgraded from bentoml 0.7, you may need to "
                        "use MultiImageHandler or LegacyImageHandler instead",
                    )
                else:
                    input_file = next(iter(files.values()))
                    input_stream = input_file.stream
                    task = InferenceTask(
                        context=InferenceContext(http_headers=req.parsed_headers),
                        data=input_stream,
                    )
            elif req.body:
                task = InferenceTask(
                    context=InferenceContext(http_headers=req.parsed_headers),
                    data=io.BytesIO(req.body),
                )
            else:
                task = InferenceTask(
                    context=InferenceContext(http_headers=req.parsed_headers),
                    data=None,
                )
                task.discard(
                    http_status=400,
                    err_msg=f'BentoML#{self.__class__.__name__} unexpected HTTP request format',
                )
            tasks[i] = task

        return tasks

    def from_aws_lambda_event(
        self, events: Iterable[AwsLambdaEvent]
    ) -> Tuple[InferenceTask[bytes]]:
        return tuple(
            InferenceTask(
                context=InferenceContext(aws_lambda_event=e),
                data=io.BytesIO(base64.decodebytes(e['body'])),
            )
            for e in events
        )

    def from_cli(self, cli_args: Tuple[str]) -> Iterator[InferenceTask[bytes]]:
        for inputs in parse_cli_inputs(cli_args):
            byte_input = inputs[0]
            yield InferenceTask(
                context=InferenceContext(cli_args=cli_args), data=io.BytesIO(byte_input)
            )

    def extract_user_func_args(
        self, tasks: Iterable[InferenceTask[bytes]]
    ) -> ApiFuncArgs:
        return (tuple(t.body for t in tasks),)
