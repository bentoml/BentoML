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

# Set BENTOML_HOME to /tmp directory due to AWS lambda disk access restrictions
os.environ['BENTOML_HOME'] = '/tmp/bentoml/'

from bentoml.deployment.aws_lambda.utils import (
    download_and_unzip_additional_packages,
)  # noqa
from bentoml.utils.s3 import download_directory_from_s3  # noqa
from bentoml.bundler import load_bento_service_class  # noqa

s3_bucket = os.environ.get('BENTOML_S3_BUCKET')
deployment_prefix = os.environ.get('BENTOML_DEPLOYMENT_PREFIX')
artifacts_prefix = os.environ.get('BENTOML_ARTIFACTS_PREFIX')
bento_name = os.environ['BENTOML_BENTO_SERVICE_NAME']
api_name = os.environ["BENTOML_API_NAME"]


bento_service_cls = load_bento_service_class(bundle_path='./{}'.format(bento_name))

# Set _bento_service_bundle_path to None, so it won't automatically load artifacts when
# we init an instance
bento_service_cls._bento_service_bundle_path = None
bento_service = bento_service_cls()

additional_pkg_dir = '/tmp/py-req'
sys.path.append(additional_pkg_dir)

if not os.path.exists(additional_pkg_dir):
    save_file_path = '/tmp/requirements.tar'
    download_and_unzip_additional_packages(
        s3_bucket, deployment_prefix, save_file_path, additional_pkg_dir
    )

# If the artifacts is not an empty list in the service, we will download them from the
# s3 bucket.
if bento_service._artifacts:
    # _load_artifacts take a directory with a subdir 'artifacts' exists
    tmp_artifacts_dir = '/tmp/bentoml/artifacts'
    os.mkdir(tmp_artifacts_dir)

    download_directory_from_s3(tmp_artifacts_dir, s3_bucket, artifacts_prefix)
    bento_service._load_artifacts('/tmp/bentoml')

this_module = sys.modules[__name__]


def api_func(event, context):  # pylint: disable=unused-argument
    api = bento_service.get_service_api(api_name)
    return api.handle_aws_lambda_event(event)


setattr(this_module, api_name, api_func)
