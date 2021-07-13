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
import json
import traceback
from typing import Iterable, Sequence, Tuple

from bentoml.adapters.string_input import StringInput
from bentoml.adapters.utils import TF_B64_KEY
from bentoml.types import InferenceTask, JsonSerializable


def b64_hook(o):
    if isinstance(o, dict) and TF_B64_KEY in o:
        return base64.b64decode(o[TF_B64_KEY])
    return o


ApiFuncArgs = Tuple[
    Sequence[JsonSerializable],
]


class TfTensorInput(StringInput):
    """
    Tensor input adapter for Tensorflow models.
    Transform incoming tf tensor data from http request, cli or lambda event into
    tf tensor.
    The behavior should be compatible with tensorflow serving REST API:
    * https://www.tensorflow.org/tfx/serving/api_rest#classify_and_regress_api
    * https://www.tensorflow.org/tfx/serving/api_rest#predict_api

    Args:
        * method: equivalence of serving API methods: (predict, classify, regress)

    Raises:
        BentoMLException: BentoML currently doesn't support Content-Type

    Examples
    --------
    Example Service:

    .. code-block:: python

        import tensorflow as tf
        import bentoml
        from bentoml.adapters import TfTensorInput
        from bentoml.frameworks.tensorflow import TensorflowSavedModelArtifact

        @bentoml.env(infer_pip_packages=True)
        @bentoml.artifacts([TensorflowSavedModelArtifact('model')])
        class MyService(bentoml.BentoService):

            @bentoml.api(input=TfTensorInput(), batch=True)
            def predict(self, input: tf.Tensor):
                result = self.artifacts.model.predict(input)
                return result

    Query with Http request:

        curl -i \\
        --header "Content-Type: application/json"
        --request POST \\
        --data '{"instances": [1]}' \\
        localhost:5000/predict


    Query with CLI command::

        bentoml run MyService:latest predict --input \\
          '{"instances": [1]}'
    """

    BATCH_MODE_SUPPORTED = True
    SINGLE_MODE_SUPPORTED = False
    METHODS = (PREDICT, CLASSIFY, REGRESS) = ("predict", "classify", "regress")

    def __init__(self, method=PREDICT, **base_kwargs):
        super().__init__(**base_kwargs)
        self.method = method

    @property
    def config(self):
        base_config = super().config
        return dict(base_config, method=self.method,)

    @property
    def request_schema(self):
        if self.method == self.PREDICT:
            return {
                "application/json": {
                    "schema": {
                        "type": "object",
                        "properties": {
                            "signature_name": {"type": "string", "default": None},
                            "instances": {
                                "type": "array",
                                "items": {"type": "object"},
                                "default": None,
                            },
                            "inputs": {"type": "object", "default": None},
                        },
                    }
                }
            }
        else:
            raise NotImplementedError(f"method {self.method} is not implemented")

    def extract_user_func_args(
        self, tasks: Iterable[InferenceTask[str]]
    ) -> ApiFuncArgs:
        import tensorflow as tf

        instances_list = []
        for task in tasks:
            try:
                parsed_json = json.loads(task.data, object_hook=b64_hook)
                if parsed_json.get("instances") is None:
                    task.discard(
                        http_status=400, err_msg="input format is not implemented",
                    )
                else:
                    instances = parsed_json.get("instances")
                    if (
                        task.http_headers.is_batch_input
                        or task.http_headers.is_batch_input is None
                    ):
                        task.batch = len(instances)
                        instances_list.extend(instances)
                    else:
                        instances_list.append(instances)
            except json.JSONDecodeError:
                task.discard(http_status=400, err_msg="Not a valid JSON format")
            except Exception:  # pylint: disable=broad-except
                err = traceback.format_exc()
                task.discard(http_status=500, err_msg=f"Internal Server Error: {err}")

        parsed_tensor = tf.constant(instances_list)
        return (parsed_tensor,)
