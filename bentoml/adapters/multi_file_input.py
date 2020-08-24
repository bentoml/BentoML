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

import io
from typing import Iterable, BinaryIO, Tuple, Iterator, Sequence, List


from bentoml.types import (
    HTTPRequest,
    AwsLambdaEvent,
    InferenceTask,
    InferenceContext,
)
from bentoml.adapters.base_input import BaseInputAdapter, parse_cli_inputs


ApiFuncArgs = Tuple[Sequence[BinaryIO], ...]
MultiFileTask = InferenceTask[Tuple[BinaryIO]]


class MultiFileInput(BaseInputAdapter[ApiFuncArgs]):
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
        from bentoml.adapters import MultiFileInput


        @bentoml.env(pip_dependencies=['torch', 'pillow', 'numpy'])
        @bentoml.artifacts([PytorchModelArtifact('classifier')])
        class PyTorchFashionClassifier(bentoml.BentoService):

            @bentoml.api(input=MultiFileInput(input_names=['image', 'label']))
            def predict(self, images, labels):
                for img_stream, label in zip(images, labels):
                    img = Image.open(img_stream)
        ```

    """

    HTTP_METHODS = ["POST"]
    BATCH_MODE_SUPPORTED = True

    def __init__(
        self, input_names, **base_kwargs,
    ):

        super().__init__(**base_kwargs)
        self.input_names = input_names

    @property
    def config(self):
        return {
            # Converting to list, google.protobuf.Struct does not work with tuple type
            "input_names": self.input_names
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

    def from_http_request(self, reqs: Iterable[HTTPRequest]) -> List[MultiFileTask]:
        tasks = []
        for req in reqs:
            if req.content_type == 'multipart/form-data':
                _, _, files = HTTPRequest.parse_form_data(req)
                try:
                    task = InferenceTask(
                        context=InferenceContext(http_headers=req.parsed_headers),
                        data=tuple(files[k] for k in self.input_names),
                    )
                except KeyError:
                    task = InferenceTask(data=None)
                    task.discard(
                        http_status=400,
                        err_msg=f"BentoML#{self.__class__.__name__} requires inputs"
                        f"fields {self.input_names}",
                    )
            else:
                task = InferenceTask(data=None)
                task.discard(
                    http_status=400,
                    err_msg=f'BentoML#{self.__class__.__name__} unexpected HTTP request'
                    ' format',
                )
            tasks.append(task)

        return tasks

    def from_aws_lambda_event(
        self, events: Iterable[AwsLambdaEvent]
    ) -> Tuple[MultiFileTask]:
        requests = tuple(
            HTTPRequest(
                headers=tuple((k, v) for k, v in e.get('headers', {}).items()),
                body=e['body'],
            )
            for e in events
        )
        return self.from_http_request(requests)

    def from_cli(self, cli_args: Tuple[str]) -> Iterator[MultiFileTask]:
        for inputs in parse_cli_inputs(cli_args, self.input_names):
            bios = []
            for input_ in inputs:
                bio = io.BytesIO(input_.read())
                bio.name = input_.name
                bios.append(bio)
            yield InferenceTask(
                context=InferenceContext(cli_args=cli_args), data=tuple(bios),
            )

    def extract_user_func_args(self, tasks: Iterable[MultiFileTask]) -> ApiFuncArgs:
        return tuple(map(tuple(zip(*tasks))))
