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

import pandas as pd
import numpy as np


class BentoHandler:
    """Handler in BentoML is the layer between a user API request and
    the input to user's API function.
    """

    def handle_request(self, request, func):
        """Handles an HTTP request, convert it into corresponding data
        format that user API function is expecting, and return API
        function result as the HTTP response to client

        :param request: Flask request object
        :param func: user API function
        """
        raise NotImplementedError

    def handle_cli(self, args, func):
        """Handles an CLI command call, convert CLI arguments into
        corresponding data format that user API function is expecting, and
        prints the API function result to console output

        :param args: CLI arguments
        :param func: user API function
        """
        raise NotImplementedError

    def handle_aws_lambda_event(self, event, func):
        """Handles a Lambda event, convert event dict into corresponding
        data format that user API function is expecting, and use API
        function result as response

        :param event: A dict containing AWS lambda event information
        :param func: user API function
        """
        raise NotImplementedError

    @property
    def request_schema(self):
        return {"application/json": {"schema": {"type": "object"}}}

    def handle_clipper_strings(self, inputs, func):
        """Handle incoming input data for clipper cluster.

        :param inputs: Incoming inputs from clipper cluster, "strings" format.
        """
        raise NotImplementedError

    def handle_clipper_bytes(self, inputs, func):
        """Handle incoming input data for clipper cluster.

        :param inputs: Incoming inputs from clipper cluster, "bytes" format.
        """
        raise NotImplementedError

    def handle_clipper_ints(self, inputs, func):
        """Handle incoming input data for clipper cluster.

        :param inputs: Incoming inputs from clipper cluster, "integers" format,
        """
        raise NotImplementedError

    def handle_clipper_doubles(self, inputs, func):
        """Handle incoming input data for clipper cluster.

        :param inputs: Incoming inputs from clipper cluster, "doubles" format,
        """
        raise NotImplementedError

    def handle_clipper_floats(self, inputs, func):
        """Handle incoming input data for clipper cluster.

        :param inputs: Incoming inputs from clipper cluster, "floats" format,
        """
        raise NotImplementedError


def get_output_str(result, output_format, output_orient="records"):
    if output_format == "str":
        return str(result)
    elif output_format == "json":
        if isinstance(result, pd.DataFrame):
            return result.to_json(orient=output_orient)
        elif isinstance(result, np.ndarray):
            return json.dumps(result.tolist())
        else:
            return json.dumps(result)
    else:
        raise ValueError("Output format {} is not supported".format(output_format))
