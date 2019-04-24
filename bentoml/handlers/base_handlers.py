# BentoML - Machine Learning Toolkit for packaging and deploying models
# Copyright (C) 2019 Atalaya Tech, Inc.

# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.

# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import json

import pandas as pd
import numpy as np


class BentoHandler():
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


def get_output_str(result, output_format, output_orient='records'):
    if output_format == 'str':
        return str(result)
    elif output_format == 'json':
        if isinstance(result, pd.DataFrame):
            return result.to_json(orient=output_orient)
        elif isinstance(result, np.ndarray):
            return json.dumps(result.tolist())
        else:
            return json.dumps(result)
    else:
        raise ValueError("Output format {} is not supported".format(output_format))
