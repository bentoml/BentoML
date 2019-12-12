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
    import unzip_requirements  # noqa # pylint: disable=unused-import
except ImportError:
    pass

# Set BENTOML_HOME to /tmp directory due to AWS lambda disk access restrictions
os.environ['BENTOML_HOME'] = '/tmp/bentoml/'

from bentoml.utils.s3 import download_directory_from_s3  # noqa
from bentoml.bundler import load_bento_service_class  # noqa

s3_bucket = os.environ.get('BENTOML_S3_BUCKET')
deployment_prefix = os.environ.get('BENTOML_DEPLOYMENT_PATH_PREFIX')
bento_name = os.environ['BENTOML_BENTO_SERVICE_NAME']
api_name = os.environ["BENTOML_API_NAME"]

bento_bundle_path = os.path.join('./', bento_name)
bento_service_cls = load_bento_service_class(bundle_path=bento_bundle_path)

# Set _bento_service_bundle_path to None, so it won't automatically load artifacts when
# we init an instance
bento_service_cls._bento_service_bundle_path = None
bento_service = bento_service_cls()

if bento_service._artifacts:
    if os.path.exists(os.path.join(bento_bundle_path, 'artifacts')):
        bento_service._load_artifacts(bento_bundle_path)
    else:
        artifact_dir = os.path.join(deployment_prefix, 'artifacts')
        os.mkdir('/tmp/bentoml/artifacts')
        download_directory_from_s3('/tmp/bentoml/artifacts', s3_bucket, artifact_dir)
        bento_service._load_artifacts('/tmp/bentoml')

this_module = sys.modules[__name__]


def api_func(event, context):  # pylint: disable=unused-argument
    api = bento_service.get_service_api(api_name)
    return api.handle_aws_lambda_event(event)


setattr(this_module, api_name, api_func)
