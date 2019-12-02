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

import json
import tensorflow as tf
from flask import make_response, Response, jsonify
from bentoml.handlers.base_handlers import BentoHandler, get_output_str


B64_KEY = 'b64'

def decode_b64_if_needed(value):
    if isinstance(value, dict):
        if B64_KEY in value:
            return base64.b64decode(value[B64_KEY])
        else:
            new_value = {}
            for k, v in value.iteritems():
                new_value[k] = decode_b64_if_needed(v)
            return new_value
    elif isinstance(value, list):
        new_value = []
        for v in value:
            new_value.append(decode_b64_if_needed(v))
        return new_value
    else:
        return value


class TensorflowTensorHandler(BentoHandler):
    """
    Tensor handlers for Tensorflow models
    ref: 
        * https://www.tensorflow.org/tfx/serving/api_rest#predict_api
        * https://github.com/tensorflow/serving/blob/91adea9716b57dd58427714427b944fbe9b3f89e/tensorflow_serving/model_servers/tensorflow_model_server_test.py#L425
    """

    def __init__(self, spec=None):
        self.spec = spec
        #self.input_names = input_names

    @property
    def request_schema(self):

        return {
            "application/json": {
                "schema": {
                    "type": "object",
                    "properties": {
                        "signature_name": {
                            "type": "string",
                            "default": None,
                        },
                        "instances": {
                            "type": "array",
                            "items": {
                                "type": "object",
                            },
                            "default": None,
                        },
                        "inputs": {
                            "type": "object",
                            "default": None,
                        }
                    },
                }
            }
        }

    def handle_request(self, request, func):
        print(func)
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
        if parsed_json.get("instances") is not None:
            instances = parsed_json.get("instances")
            instances = decode_b64_if_needed(instances)

            if self.spec is not None:
                parsed_tensor = tf.constant(instances, self.spec.dtype)
                # origin_shape_map = {parsed_tensor._id: parsed_tensor.shape}
                if not self.spec.is_compatible_with(parsed_tensor):
                    parsed_tensor = tf.reshape(parsed_tensor, tuple(i is None and -1 or i for i in self.spec.shape))
                result = func(parsed_tensor)
                # if result._id in origin_shape_map:
                #     result = tf.reshape(result, origin_shape_map.get(result._id))
                if isinstance(result, tf.Tensor):
                    result = result.numpy().tolist()
            else:
                parsed_tensor = tf.constant(instances)
                result = func(parsed_tensor)

        elif parsed_json.get("inputs"):
            # column mode
            raise NotImplementedError

        output_format = request.headers.get("output", "json")
        if output_format == "json":
            result_object = {"predictions": result}
            result_str = get_output_str(result_object, output_format)
        elif output_format == "str":
            result_str = get_output_str(result, output_format)
        else:
            return make_response(
                jsonify(
                    message="Request output must be 'json' or 'str'"
                    "for this BentoService API"
                ),
                400,
            )

        return Response(response=result_str, status=200, mimetype="application/json")

    def handle_cli(self, args, func):
        raise NotImplementedError

    def handle_aws_lambda_event(self, event, func):
        raise NotImplementedError
