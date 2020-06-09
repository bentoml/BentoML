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
import io
import os
import sys
import tarfile
import logging

import boto3
from botocore.exceptions import ClientError

logger = logging.getLogger(__name__)


def download_extra_resources():
    s3_bucket = os.environ.get('BENTOML_S3_BUCKET')
    s3_prefix = os.environ.get('BENTOML_DEPLOYMENT_PATH_PREFIX')
    additional_pkg_dir = '/tmp/requirements'

    if not os.path.exists(additional_pkg_dir) or not os.listdir(additional_pkg_dir):
        # Using print instead of logger.info, because this ran before bentoml is loaded.
        print('Additional required modules are not present. Downloading from s3')
        s3_file_path = os.path.join(s3_prefix, 'requirements.tar')

        s3_client = boto3.client('s3')
        print(f'requirement.tar does not exist, downloading from {s3_bucket}')
        try:
            obj = s3_client.get_object(Bucket=s3_bucket, Key=s3_file_path)
            bytestream = io.BytesIO(obj['Body'].read())
            print('Extracting required modules from tar file')
            tar = tarfile.open(fileobj=bytestream, mode='r:*')
            tar.extractall(path='/tmp')
            print('Appending /tmp/requirements to PYTHONPATH')
            sys.path.append(additional_pkg_dir)
        except ClientError as e:
            if e.response['Error']['Code'] == "404":
                print(f'File {s3_bucket}/{s3_file_path} does not exists')
                logger.error('File %s/%s does not exists', s3_bucket, s3_file_path)
            else:
                raise
    else:
        print('Additional required modules already present, skipping download from s3')
