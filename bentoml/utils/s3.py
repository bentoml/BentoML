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
import logging
from urllib.parse import urlparse

import boto3
from botocore.exceptions import ClientError

from bentoml.exceptions import BentoMLException

logger = logging.getLogger(__name__)


def is_s3_url(url):
    """
    Check if url is an s3, s3n, or s3a url
    """
    try:
        return urlparse(url).scheme in ["s3", "s3n", "s3a"]
    except Exception:  # pylint:disable=broad-except
        return False


def upload_directory_to_s3(
    upload_directory_path, region, bucket_name, s3_path_prefix=''
):
    s3_client = boto3.client('s3', region)
    try:
        for root, _, file_names in os.walk(upload_directory_path):
            relative_path_to_upload_dir = os.path.relpath(root, upload_directory_path)
            if relative_path_to_upload_dir == '.':
                relative_path_to_upload_dir = ''
            for file_name in file_names:
                key = os.path.join(
                    s3_path_prefix, relative_path_to_upload_dir, file_name
                )
                logger.debug(
                    'Uploading %s to s3 %s', file_name, bucket_name + '/' + key
                )
                s3_client.upload_file(os.path.join(root, file_name), bucket_name, key)
    except ClientError as error:
        raise BentoMLException(
            "Failed to upload directory to s3 bucket {}, error: {}".format(
                bucket_name, str(error)
            )
        )


def create_s3_bucket_if_not_exists(bucket_name, region):
    s3_client = boto3.client('s3', region)
    try:
        s3_client.get_bucket_acl(Bucket=bucket_name)
        logger.debug("Found bucket %s in region %s already exist", bucket_name, region)
    except ClientError as error:
        if error.response and error.response['Error']['Code'] == 'NoSuchBucket':
            logger.debug('Creating s3 bucket: %s in region %s', bucket_name, region)
            s3_client.create_bucket(
                Bucket=bucket_name,
                CreateBucketConfiguration={'LocationConstraint': region},
            )
        else:
            raise error


def is_s3_bucket_exist(bucket_name, region):
    s3_client = boto3.client('s3', region)
    try:
        s3_client.get_bucket_acl(Bucket=bucket_name)
        return True
    except ClientError as error:
        if error.response and error.response['Error']['Code'] == 'NoSuchBucket':
            return False
        else:
            raise error


def download_directory_from_s3(download_dest_directory, s3_bucket, path_prefix):
    """ Download directory from s3 bucket to given directory.
    Args:
        download_dest_directory: String
        s3_bucket: String
        path_prefix: String

    Returns: None
    """
    s3_client = boto3.client('s3')
    try:
        list_content_result = s3_client.list_objects(
            Bucket=s3_bucket, Prefix=path_prefix
        )
        for content in list_content_result['Contents']:
            file_name = content['Key'].split('/')[-1]
            file_path = os.path.join(download_dest_directory, file_name)
            if not os.path.isfile(file_path):
                s3_client.download_file(s3_bucket, content['Key'], file_path)
            else:
                logger.error('File %s already exists', file_path)
    except ClientError as e:
        logger.error('Error getting object from bucket %s, %s', s3_bucket, e)
        raise e
