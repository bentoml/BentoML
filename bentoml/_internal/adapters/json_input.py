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

import json
import traceback
from typing import Iterable, Sequence, Tuple

from bentoml.adapters.string_input import StringInput
from bentoml.types import InferenceTask, JsonSerializable

ApiFuncArgs = Tuple[
    Sequence[JsonSerializable],
]


class JsonInput(StringInput):
    """JsonInput parses REST API request or CLI command into parsed_jsons(a list of
    json serializable object in python) and pass down to user defined API function

    Examples
    ----------
    Example services:

    Use JsonInput with ``batch=True``

    .. code-block:: python

        from typing import List
        from bentoml.types import JsonSerializable

        @bentoml.api(input=JsonInput(), batch=True)
        def predict(self, parsed_json_list: List[JsonSerializable]):
            results = self.artifacts.classifier([j['text'] for j in parsed_json_list])
            return results

    OR use JsonInput with ``batch=False`` (the default)

    .. code-block:: python

        @bentoml.api(input=JsonInput())
        def predict(self, parsed_json):
            results = self.artifacts.classifier([parsed_json['text']])
            return results[0]

    For client prediction request, it is the same for both batch and non-batch API,
    the request should contain only one single input item::

        curl -i \\
          --header "Content-Type: application/json" \\
          --request POST \\
          --data '{"text": "best movie ever"}' \\
          localhost:5000/predict
    """

    def extract_user_func_args(
        self, tasks: Iterable[InferenceTask[str]]
    ) -> ApiFuncArgs:
        json_inputs = []
        for task in tasks:
            try:
                parsed_json = json.loads(task.data)
                json_inputs.append(parsed_json)
            except json.JSONDecodeError:
                task.discard(http_status=400, err_msg="Not a valid JSON format")
            except Exception:  # pylint: disable=broad-except
                err = traceback.format_exc()
                task.discard(http_status=500, err_msg=f"Internal Server Error: {err}")
        return (json_inputs,)
