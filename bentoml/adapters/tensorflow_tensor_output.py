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
import itertools
from typing import Iterable

import numpy as np
import argparse

from bentoml.marshal.utils import SimpleResponse, SimpleRequest
from bentoml.adapters.base_output import BaseOutputAdapter


TF_B64_KEY = "b64"


def to_numpy(tensor):
    '''
    Tensor -> ndarray
    List[Tensor] -> tuple[ndarray]
    '''
    import tensorflow as tf

    if isinstance(tensor, (list, tuple)):
        return tuple(to_numpy(t) for t in tensor)

    if tf.__version__.startswith("1."):
        with tf.compat.v1.Session():
            return tensor.numpy()
    else:
        return tensor.numpy()


class TfTensorJsonEncoder(json.JSONEncoder):
    """ Special json encoder for numpy types """

    def default(self, o):  # pylint: disable=method-hidden

        import tensorflow as tf

        # Tensor -> ndarray or object
        if isinstance(o, tf.Tensor):
            if tf.__version__.startswith("1."):
                with tf.compat.v1.Session():
                    return o.numpy()
            else:
                return o.numpy()
        if isinstance(o, np.generic):
            return o.item()
        if isinstance(o, np.ndarray):
            return o.tolist()
        if isinstance(o, bytes):
            try:
                return o.decode('utf-8')
            except UnicodeDecodeError:
                return {TF_B64_KEY: base64.b64encode(o).decode("utf-8")}
        return json.JSONEncoder.default(self, o)


def jsonize(result, indent=None):
    try:
        return json.dumps(result, cls=TfTensorJsonEncoder, indent=indent)
    except (TypeError, OverflowError):
        # when result is not JSON serializable
        return json.dumps({"result": str(result)})


class TfTensorOutput(BaseOutputAdapter):
    """
    Converts result of use defined API function into specific output.

    Args:
        cors (str): The value of the Access-Control-Allow-Origin header set in the
            AWS Lambda response object. Default is "*". If set to None,
            the header will not be set.
    """

    def __init__(self, **kwargs):
        super(TfTensorOutput, self).__init__(**kwargs)

    def to_batch_response(
        self,
        result_conc,
        slices=None,
        fallbacks=None,
        requests: Iterable[SimpleRequest] = None,
    ) -> Iterable[SimpleResponse]:
        # TODO(bojiang): header content_type
        results = to_numpy(result_conc)
        assert isinstance(results, np.ndarray)
        if slices is None:
            slices = [i for i in range(results.shape[0])]
        if fallbacks is None:
            fallbacks = itertools.repeat(None)
        responses = [None] * len(slices)

        for i, (s, f, r) in enumerate(zip(slices, fallbacks, requests)):
            if s is None:
                responses[i] = f
                continue
            batch_flag = self.is_batch_request(r)
            result = results[s]
            if batch_flag:
                result_str = jsonize(result[0])
            else:
                result_str = jsonize(result)
            responses[i] = SimpleResponse(200, dict(), result_str)
        return responses

    def to_cli(self, result, args):
        """
        Handles an CLI command call, convert CLI arguments into
        corresponding data format that user API function is expecting, and
        prints the API function result to console output

        :param args: CLI arguments
        """
        parser = argparse.ArgumentParser()
        parser.add_argument("-o", "--output", default="json", choices=["str", "json"])
        parsed_args = parser.parse_args(args)
        if parsed_args.output == 'json':
            result = jsonize(result, indent='  ')
        else:
            result = str(result)
        print(result)

    def to_aws_lambda_event(self, result, event):
        result = jsonize(result)
        # Allow disabling CORS by setting it to None
        if self.cors:
            return {
                "statusCode": 200,
                "body": result,
                "headers": {"Access-Control-Allow-Origin": self.cors},
            }
        return {"statusCode": 200, "body": result}

    @property
    def pip_dependencies(self):
        """
        :return: List of PyPI package names required by this OutputAdapter
        """
        return ['tensorflow']
