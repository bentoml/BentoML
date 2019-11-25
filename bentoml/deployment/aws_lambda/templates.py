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


AWS_LAMBDA_APP_PY_TEMPLATE = """\
import os
import sys

# Set BENTOML_HOME to /tmp directory due to AWS lambda disk access restrictions
os.environ['BENTOML_HOME'] = '/tmp/bentoml/'
from bentoml.deployment.aws_lambda.utils import download_artifacts_for_lambda_function
from bentoml.bundler import load_bento_service_class

bento_name = os.environ.get('BENTO_SERVICE_NAME')
s3_bucket = os.environ.get('S3_BUCKET')
artifacts_prefix = os.environ.get('ARTIFACTS_PREFIX')
tmp_artifacts_dir = '/tmp/bentoml'

download_artifacts_for_lambda_function(tmp_artifacts_dir, s3_bucket, artifacts_prefix)

bento_service_cls = load_bento_service_class(bundle_path='./{}'.format(bento_name))
# Set _bento_service_bundle_path to None, so it won't automatically load artifacts when
# we init an instance
bento_service_cls._bento_service_bundle_path = None
bento_service = bento_service_cls()
bento_service._load_artifacts('/tmp/bentoml')

this_module = sys.modules[__name__]
api_name = os.environ.get("BENTOML_API_NAME")

def api_func(event, context):
    api = bento_service.get_service_api(api_name)
    return api.handle_aws_lambda_event(event)

setattr(this_module, api_name, api_func)
"""
