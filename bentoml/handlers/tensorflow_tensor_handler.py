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

from bentoml.handlers.base_handlers import BentoHandler


class TensorflowTensorHandler(BentoHandler):
    """
    Tensor handlers for Tensorflow models
    ref: 
        * https://www.tensorflow.org/tfx/serving/api_rest#predict_api
        * https://github.com/tensorflow/serving/blob/91adea9716b57dd58427714427b944fbe9b3f89e/tensorflow_serving/model_servers/tensorflow_model_server_test.py#L425
    """

    @property
    def request_schema(self):
        default = {"application/json": {"schema": {"type": "object"}}}
        if self.input_dtypes is None:
            return default

        if isinstance(self.input_dtypes, dict):
            return {
                "application/json": {  # For now, only declare JSON on docs.
                    "schema": {
                        "type": "object",
                        "properties": {
                            k: {"type": "array", "items": {"type": self._get_type(v)}}
                            for k, v in self.input_dtypes.items()
                        },
                    }
                }
            }

        return default

    def handle_request(self, request, func):
        if request.content_type == "application/json":
            parsed_json = json.loads(request.data.decode("utf-8"))
        else:
            return make_response(
                jsonify(
                    message="Request content-type must be 'application/json'"
                    "for this BentoService API"
                ),
                400,
            )
        result = func(parsed_json)
        result = get_output_str(result, request.headers.get("output", "json"))
        return Response(response=result, status=200, mimetype="application/json")

    def handle_cli(self, args, func):
        raise NotImplementedError

    def handle_aws_lambda_event(self, event, func):
        raise NotImplementedError
