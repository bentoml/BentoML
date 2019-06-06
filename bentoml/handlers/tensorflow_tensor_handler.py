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
    """

    def handle_request(self, request, func):
        raise NotImplementedError

    def handle_cli(self, args, func):
        raise NotImplementedError

    def handle_aws_lambda_event(self, event, func):
        raise NotImplementedError
