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

import os
import sys

try:
    import download_extra_resources

    download_extra_resources.download_extra_resources()
except ImportError:
    # When function doesn't have extra resources or dependencies, we will not include
    # unzip_extra_resources and that will result with ImportError.  We will let it fail
    # silently.
    pass

# Set BENTOML_HOME to /tmp directory due to AWS lambda disk access restrictions
os.environ['BENTOML_HOME'] = '/tmp/bentoml/'
from bentoml import load  # noqa

bento_name = os.environ['BENTOML_BENTO_SERVICE_NAME']
api_name = os.environ["BENTOML_API_NAME"]

bento_bundle_path = os.path.join('./', bento_name)
if not os.path.exists(bento_bundle_path):
    bento_bundle_path = os.path.join('/tmp/requirements', bento_name)

print('Bento bundle path is', bento_bundle_path)
bento_service = load(bento_bundle_path)

print('Bento service loaded', bento_service)
this_module = sys.modules[__name__]


def api_func(event, context):  # pylint: disable=unused-argument
    api = bento_service.get_service_api(api_name)
    return api.handle_aws_lambda_event(event)


setattr(this_module, api_name, api_func)
