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
import boto3
from six.moves.urllib.parse import urlparse

from bentoml.utils import Path


def is_s3_url(url):
    """
    Check if url is an s3, s3n, or s3a url
    """
    try:
        return urlparse(url).scheme in ["s3", "s3n", "s3a"]
    except Exception:  # pylint:disable=broad-except
        return False


def upload_to_s3(s3_url, file_path):
    """
    Update files in the file_path to the s3 location
    """

    parse_result = urlparse(s3_url)
    bucket = parse_result.netloc
    base_path = parse_result.path

    s3_client = boto3.client("s3")

    for root, _, files in os.walk(file_path):
        for file_name in files:
            abs_file_path = os.path.join(root, file_name)
            relative_file_path = abs_file_path[len(file_path) + 1 :]  # noqa: E203
            s3_path = os.path.join(base_path, relative_file_path)
            s3_client.upload_file(Filename=abs_file_path, Bucket=bucket, Key=s3_path)


def download_from_s3(s3_url, file_path):
    """
    Download files from given s3_path and store in the given file path
    """
    parse_result = urlparse(s3_url)
    bucket = parse_result.netloc
    base_path = parse_result.path

    s3_client = boto3.client("s3")
    list_object_result = s3_client.list_objects(Bucket=bucket, Prefix=base_path)
    result_content = list_object_result["Contents"]

    for content in result_content:
        relative_file_path = content["Key"][len(base_path) + 1 :]  # noqa: E203
        local_file_path = os.path.join(file_path, relative_file_path)
        Path(os.path.dirname(local_file_path)).mkdir(parents=True, exist_ok=True)
        s3_client.download_file(
            Bucket=bucket, Key=content["Key"], Filename=local_file_path
        )
