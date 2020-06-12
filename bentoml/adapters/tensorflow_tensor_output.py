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

from typing import Iterable

import argparse

from bentoml.marshal.utils import SimpleResponse, SimpleRequest
from bentoml.adapters.base_output import BaseOutputAdapter
from bentoml.adapters.json_output import jsonize
from bentoml.adapters.utils import NestedConverter, tf_tensor_2_serializable


decode_tf_if_needed = NestedConverter(tf_tensor_2_serializable)


class TfTensorOutput(BaseOutputAdapter):
    """
    Converts result of use defined API function into specific output.

    Args:
        output_orient (str): Prefer json orient format for output result. Default is
            records.
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
        result_conc = decode_tf_if_needed(result_conc)

        assert isinstance(result_conc, (list, tuple))

        if slices is None:
            slices = [i for i, _ in enumerate(result_conc)]
        if fallbacks is None:
            fallbacks = [None] * len(slices)

        responses = [None] * len(slices)

        for i, (s, f) in enumerate(zip(slices, fallbacks)):
            if s is None:
                responses[i] = f
                continue
            result = result_conc[s]
            result_str = jsonize(result)
            responses[i] = SimpleResponse(200, dict(), result_str)

        return responses

    def to_cli(self, result, args):
        """Handles an CLI command call, convert CLI arguments into
        corresponding data format that user API function is expecting, and
        prints the API function result to console output

        :param args: CLI arguments
        """
        parser = argparse.ArgumentParser()
        parser.add_argument("-o", "--output", default="str", choices=["str", "json"])
        parsed_args = parser.parse_args(args)

        result = decode_tf_if_needed(result)
        if parsed_args.output == 'json':
            result = jsonize(result)
        else:
            result = str(result)
        print(result)

    def to_aws_lambda_event(self, result, event):

        result = decode_tf_if_needed(result)
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
