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
import base64
import argparse
import tensorflow as tf
import numpy as np
from flask import make_response, Response, jsonify
from bentoml.handlers.base_handlers import BentoHandler, get_output_str
from bentoml.exceptions import BentoMLException


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
    Tensor handlers for Tensorflow models.
    Transform incoming tf tensor data from http request, cli or lambda event into
    tf tensor.
    The behaviour should be compatible with tensorflow serving REST predict API:
        * https://www.tensorflow.org/tfx/serving/api_rest#predict_api

    Args:
        -

    Raises:
        BentoMLException: BentoML currently doesn't support Content-Type
    """

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

    def _handle_raw_str(self, raw_str, output_format, func):
        parsed_json = json.loads(raw_str)
        if parsed_json.get("instances") is not None:
            instances = parsed_json.get("instances")
            instances = decode_b64_if_needed(instances)
            parsed_tensor = tf.constant(instances)
            result = func(parsed_tensor)
            if isinstance(result, tf.Tensor):
                result = result.numpy().tolist()
            if isinstance(result, np.ndarray):
                result = result.tolist()

        elif parsed_json.get("inputs"):
            # column mode
            raise NotImplementedError

        if output_format == "json":
            result_object = {"predictions": result}
            result_str = get_output_str(result_object, output_format)
        elif output_format == "str":
            result_str = get_output_str(result, output_format)

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
        output_format = request.headers.get("output", "json")
        if output_format not in {"json", "str"}:
            return make_response(
                jsonify(
                    message="Request output must be 'json' or 'str'"
                    "for this BentoService API"
                ),
                400,
            )
        if request.content_type == "application/json":
            input_str = request.data.decode("utf-8")
            output_format = request.headers.get("output", "json")
            result_str = self._handle_raw_str(input_str, output_format, func)
            return Response(
                response=result_str, status=200, mimetype="application/json")
        else:
            return make_response(
                jsonify(
                    message="Request content-type must be 'application/json'"
                    "for this BentoService API"
                ),
                400,
            )

    def handle_cli(self, args, func):
        parser = argparse.ArgumentParser()
        parser.add_argument("--input", required=True)
        parser.add_argument(
            "-o", "--output", default="str", choices=["str", "json"]
        )
        parsed_args = parser.parse_args(args)

        result = self._handle_raw_str(parsed_args.input, parsed_args.output, func)
        print(result)

    def handle_aws_lambda_event(self, event, func):
        if event["headers"].get("Content-Type", "") == "application/json":
            result = self._handle_raw_str(
                event["body"], event["headers"].get("output", "json"), func)
        else:
            raise BentoMLException(
                "BentoML currently doesn't support Content-Type: {content_type} for "
                "AWS Lambda".format(content_type=event["headers"]["Content-Type"])
            )

        return {"statusCode": 200, "body": result}
