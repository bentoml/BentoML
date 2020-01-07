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
import argparse
from flask import Response
from bentoml.handlers.utils import (
    NestedConverter,
    tf_b64_2_bytes,
    tf_tensor_2_serializable,
)
from bentoml.handlers.base_handlers import BentoHandler, api_func_result_to_json
from bentoml.exceptions import BentoMLException, BadInput


decode_b64_if_needed = NestedConverter(tf_b64_2_bytes)
decode_tf_if_needed = NestedConverter(tf_tensor_2_serializable)


class TensorflowTensorHandler(BentoHandler):
    """
    Tensor handlers for Tensorflow models.
    Transform incoming tf tensor data from http request, cli or lambda event into
    tf tensor.
    The behaviour should be compatible with tensorflow serving REST API:
    * https://www.tensorflow.org/tfx/serving/api_rest#classify_and_regress_api
    * https://www.tensorflow.org/tfx/serving/api_rest#predict_api

    Args:
        * method: equivalence of serving API methods: (predict, classify, regress)

    Raises:
        BentoMLException: BentoML currently doesn't support Content-Type
    """

    METHODS = (PREDICT, CLASSIFY, REGRESS) = ("predict", "classify", "regress")

    def __init__(self, method=PREDICT):
        self.method = method

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

    def _handle_raw_str(self, raw_str, output_format, func):
        import tensorflow as tf

        parsed_json = json.loads(raw_str)
        if parsed_json.get("instances") is not None:
            instances = parsed_json.get("instances")
            instances = decode_b64_if_needed(instances)
            parsed_tensor = tf.constant(instances)
            result = func(parsed_tensor)
            result = decode_tf_if_needed(result)

        elif parsed_json.get("inputs"):
            raise NotImplementedError("column format 'inputs' is not implemented")

        if output_format == "json":
            result_str = api_func_result_to_json(result)
        elif output_format == "str":
            result_str = str(result)

        return result_str

    def handle_request(self, request, func):
        """Handle http request that has jsonlized tensorflow tensor. It will convert it
        into a tf tensor for the function to consume.

        Args:
            request: incoming request object.
            func: function that will take ndarray as its arg.
        Return:
            response object
        """
        if request.content_type == "application/json":
            input_str = request.data.decode("utf-8")
            result_str = self._handle_raw_str(input_str, "json", func)
            return Response(
                response=result_str, status=200, mimetype="application/json"
            )
        else:
            raise BadInput(
                "Request content-type must be 'application/json'"
                " for this BentoService API"
            )

    def handle_cli(self, args, func):
        parser = argparse.ArgumentParser()
        parser.add_argument("--input", required=True)
        parser.add_argument("-o", "--output", default="str", choices=["str", "json"])
        parsed_args = parser.parse_args(args)

        result = self._handle_raw_str(parsed_args.input, parsed_args.output, func)
        print(result)

    def handle_aws_lambda_event(self, event, func):
        if event["headers"].get("Content-Type", "") == "application/json":
            result = self._handle_raw_str(event["body"], "json", func)
        else:
            raise BentoMLException(
                "BentoML currently doesn't support Content-Type: {content_type} for "
                "AWS Lambda".format(content_type=event["headers"]["Content-Type"])
            )

        return {"statusCode": 200, "body": result}
