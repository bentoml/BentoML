# BentoML - Machine Learning Toolkit for packaging and deploying models
# Copyright (C) 2019 Atalaya Tech, Inc.

# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.

# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

import os
import boto3
from six.moves.urllib.parse import urlparse


def check_is_s3_path(path):
    """
    This helper function return True if path is in s3:// format
    """
    parsed_url = urlparse(path)
    if parsed_url.scheme == 's3':
        return True
    return False


def find_bucket_key(s3_path):
    """
    This is a helper function that given an s3 path such that the path is of
    the form: bucket/key
    It will return the bucket and the key represented by the s3 path
    """
    s3_components = s3_path.split('/')
    bucket = s3_components[0]
    s3_key = ""
    if len(s3_components) > 1:
        s3_key = '/'.join(s3_components[1:])
    return bucket, s3_key


def split_s3_bucket_key(s3_path):
    """Split s3 path into bucket and key prefix.
    This will also handle the s3:// prefix.
    :return: Tuple of ('bucketname', 'keyname')
    """
    if s3_path.startswith('s3://'):
        s3_path = s3_path[5:]
    return find_bucket_key(s3_path)


def create_bucket_if_not_exit(client, bucket):
    """
    Create s3 bucket with the giving bucket name, if it didn't exist
    """
    try:
        client.head_bucket(Bucket=bucket)
    except Exception:
        region_name = boto3.Session().region_name
        client.create_bucket(
            Bucket=bucket,
            CreateBucketConfiguration={'LocationConstraint': region_name}
        )


def upload_to_s3(s3_path, file_path):
    """
    Update files in the file_path to the s3 location
    """
    bucket, s3_key = split_s3_bucket_key(s3_path)

    try:
        s3_client = boto3.client('s3')
        create_bucket_if_not_exit(s3_client, bucket)

        for root, dirs, files in os.walk(file_path):
            for file_name in files:
                submit_file_path = os.path.join(root, file_name)
                full_path = submit_file_path[len(file_path) + 1:]
                s3_path = os.path.join(s3_key, full_path)
                s3_client.upload_file(Filename=submit_file_path, Bucket=bucket, Key=s3_path)

        #TODO: Clean up files in the temp dir
    except Exception as e:
        raise e


def download_from_s3(s3_path, file_path):
    """
    Download files from given s3_path and store in the given file path
    """
    bucket, s3_key = split_s3_bucket_key(s3_path)

    try:
        s3_client = boto3.client('s3')
        list_object_result = s3_client.list_objects(Bucket=bucket, Prefix=s3_key)
        result_content = list_object_result['Contents']

        for content in result_content:
            full_path = content['Key'][len(s3_key) + 1:]
            local_file_path = os.path.join(file_path, full_path)
            if not os.path.exists(os.path.dirname(local_file_path)):
                os.makedirs(os.path.dirname(local_file_path))
            s3_client.download_file(Bucket=bucket, Key=content['Key'], Filename=local_file_path)

        return file_path
    except Exception as e:
        raise e
