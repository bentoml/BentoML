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

DEFAULT_CLIPPER_ENTRY = """\
from __future__ import print_function

import rpc # this is copied from clipper
import os
import sys

from bentoml import load

IMPORT_ERROR_RETURN_CODE = 3

bento_service = load('/bento_model')
apis = bento_service.get_service_apis()

api = next(item for item in apis if item.name == '{api_name}')
if not api:
    raise BentoMLException("Can't find api with name %s" % {api_name})

class BentoClipperContainer(rpc.ModelContainerBase):
    def __init__(self):
        self.input_type = '{input_type}'

    def predict_ints(self, inputs):
        preds = api.handle_clipper_numbers(inputs)
        return [str(p) for p in preds]

    def predict_floats(self, inputs):
        preds = api.handle_clipper_numbers(inputs)
        return [str(p) for p in preds]

    def predict_doubles(self, inputs):
        preds = api.handle_clipper_numbers(inputs)
        return [str(p) for p in preds]

    def predict_bytes(self, inputs):
        preds = api.handle_clipper_bytes(inputs)
        return [str(p) for p in preds]

    def predict_(self, inputs):
        preds = api.handle_clipper_strings(inputs)
        return [str(p) for p in preds]


if __name__ == "__main__":
    print("Starting Bento service Clipper Containter")
    rpc_service = rpc.RPCService()
    
    try:
        model = BentoClipperContainer()
        sys.stdout.flush()
        sys.stderr.flush()
    except ImportError:
        sys.exit(IMPORT_ERROR_RETURN_CODE)

    rpc_service.start(model)
"""
