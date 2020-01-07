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

    print("Downloading extra requirements and files from s3..")
    download_extra_resources.download_extra_resources()
    print("Finished downloading extra requirements and files")
except ImportError:
    # When function doesn't have extra resources or dependencies, we will not include
    # unzip_extra_resources and that will result with ImportError.  We will let it fail
    # silently
    pass

# Set BENTOML_HOME to /tmp directory due to AWS lambda disk access restrictions
os.environ['BENTOML_HOME'] = '/tmp/bentoml/'
from bentoml import load  # noqa

bento_name = os.environ['BENTOML_BENTO_SERVICE_NAME']
api_name = os.environ["BENTOML_API_NAME"]

bento_bundle_path = os.path.join('./', bento_name)
if not os.path.exists(bento_bundle_path):
    bento_bundle_path = os.path.join('/tmp/requirements', bento_name)

print(f'Loading BentoService bundle from path: "{bento_bundle_path}"')
bento_service = load(bento_bundle_path)
print(f'BentoService "{bento_service.name}" loaded successfully')
bento_service_api = bento_service.get_service_api(api_name)
print(f'BentoService API "{api_name}" loaded successfully')

this_module = sys.modules[__name__]


def api_func(event, context):  # pylint: disable=unused-argument
    """
    Event – AWS Lambda uses this parameter to pass in event data to the handler. This
    parameter is usually of the Python dict type. It can also be list, str, int,
    float, or NoneType type. Currently BentoML only handles Lambda event coming from
    Application Load Balancer, in which case the parameter `event` must be type `dict`
    containing the HTTP request headers and body.

    see: https://docs.aws.amazon.com/lambda/latest/dg/services-alb.html
    """

    if type(event) is dict and "headers" in event and "body" in event:
        return bento_service_api.handle_aws_lambda_event(event)
    else:
        error_msg = (
            'Error: Unexpected Lambda event data received. Currently BentoML lambda '
            'deployment can only handle event triggered by HTTP request from '
            'Application Load Balancer.'
        )
        print(error_msg)
        raise RuntimeError(error_msg)


setattr(this_module, api_name, api_func)
