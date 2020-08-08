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

import os
import argparse
import base64
import contextlib
from io import BytesIO
from typing import Iterable

from werkzeug.wrappers import Request

from bentoml.exceptions import BadInput
from bentoml.adapters.base_input import BaseInputAdapter
from bentoml.marshal.utils import SimpleResponse, SimpleRequest


class FileInput(BaseInputAdapter):
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

    def __init__(
        self, **base_kwargs,
    ):
        super(FileInput, self).__init__(**base_kwargs)

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

    def _load_file(self, request: Request):
        if len(request.files):
            if len(request.files) != 1:
                raise BadInput(
                    "ImageHandler requires one and at least one file at a time, "
                    "if you just upgraded from bentoml 0.7, you may need to use "
                    "MultiImageHandler or LegacyImageHandler instead"
                )
            input_file = next(iter(request.files.values()))
            if not input_file:
                raise BadInput("BentoML#ImageHandler unexpected HTTP request format")
            input_stream = input_file.stream
        else:
            data = request.get_data()
            if not data:
                raise BadInput("BentoML#ImageHandler unexpected HTTP request format")
            else:
                input_stream = BytesIO(data)

        return input_stream

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
                input_data = self._load_file(request)
            except BadInput:
                ids.append(None)
                continue

            input_datas.append(input_data)
            ids.append(i)

        results = func(input_datas) if input_datas else []
        return self.output_adapter.to_batch_response(results, ids, requests)

    def handle_request(self, request, func):
        """Handle http request that has one file. It will convert file into a
        BytesIO object for the function to consume.

        Args:
            request: incoming request object.
            func: function that will take ndarray as its arg.
        Return:
            response object
        """
        input_data = self._load_file(request)
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
            input_list = []
            with contextlib.ExitStack() as stack:
                for file_path in step_file_paths:
                    if not os.path.isabs(file_path):
                        file_path = os.path.abspath(file_path)
                    input_list.append(stack.enter_context(open(file_path, 'rb')))

                results = func(input_list)

            for result in results:
                self.output_adapter.to_cli(result, unknown_args)

    def handle_aws_lambda_event(self, event, func):
        f = base64.decodebytes(event["body"])
        result = func((BytesIO(f),))[0]
        return self.output_adapter.to_aws_lambda_event(result, event)
