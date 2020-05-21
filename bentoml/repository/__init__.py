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
import shutil
import boto3
import logging
from pathlib import Path
from abc import abstractmethod, ABCMeta
from urllib.parse import urlparse

from botocore.exceptions import ClientError

from bentoml import config
from bentoml.exceptions import YataiRepositoryException
from bentoml.utils.s3 import is_s3_url
from bentoml.proto.repository_pb2 import BentoUri


logger = logging.getLogger(__name__)


class BentoRepositoryBase(object):
    """
    BentoRepository is the interface for managing saved Bentos over file system or
    cloud storage systems.

    A Bento is a BentoService serialized into a standard file format that can be
    easily load back to a Python session, installed as PyPI package, or run in Conda
    or docker environment with all dependencies configured automatically
    """

    __metaclass__ = ABCMeta

    @abstractmethod
    def add(self, bento_name, bento_version):
        """
        Proposing to add a saved BentoService to target repository by providing the
        bento name and version.
        Return value is an URL(file path or s3 path for example), that is ready for
        the client to upload saved Bento files.
        """

    @abstractmethod
    def get(self, bento_name, bento_version):
        """
        Get a file path to the saved Bento files, path must be accessible form local
        machine either through NFS or pre-downloaded to local machine
        """

    @abstractmethod
    def dangerously_delete(self, bento_name, bento_version):
        """
        Deleting the Bento files that was added to this repository earlier, this may
        break existing deployments or create issues when doing deployment rollback
        """


class _LocalBentoRepository(BentoRepositoryBase):
    def __init__(self, base_url):
        if not os.path.exists(base_url):
            # make sure local repo base path exist
            os.mkdir(base_url)

        self.base_path = base_url
        self.uri_type = BentoUri.LOCAL

    def add(self, bento_name, bento_version):
        # Full path containing saved BentoService bundle, it the base path with service
        # name and service version as prefix. e.g.:
        # with base_path = '/tmp/my_bento_repo/', the saved bento will resolve in
        # the directory: '/tmp/my_bento_repo/service_name/version/'
        target_dir = os.path.join(self.base_path, bento_name, bento_version)

        # Ensure parent directory exist
        Path(os.path.join(self.base_path), bento_name).mkdir(
            parents=True, exist_ok=True
        )

        # Raise if target bento version already exist in storage
        if os.path.exists(target_dir):
            raise YataiRepositoryException(
                "Existing BentoService bundle {name}:{version} found in repository: "
                "{target_dir}".format(
                    name=bento_name, version=bento_version, target_dir=target_dir
                )
            )

        # Create target directory for upload
        os.mkdir(target_dir)

        return BentoUri(type=self.uri_type, uri=target_dir)

    def get(self, bento_name, bento_version):
        saved_path = os.path.join(self.base_path, bento_name, bento_version)
        if not os.path.exists(saved_path):
            raise YataiRepositoryException(
                "Bento {}:{} not found in target repository".format(
                    bento_name, bento_version
                )
            )
        return saved_path

    def dangerously_delete(self, bento_name, bento_version):
        saved_path = os.path.join(self.base_path, bento_name, bento_version)
        try:
            return shutil.rmtree(saved_path)
        except FileNotFoundError:
            logger.warning(
                "BentoService %s:%s has already been deleted from local storage",
                bento_name,
                bento_version,
            )
            return


class _S3BentoRepository(BentoRepositoryBase):
    def __init__(self, base_url, s3_endpoint_url=None):
        self.uri_type = BentoUri.S3

        parse_result = urlparse(base_url)
        self.bucket = parse_result.netloc
        self.base_path = parse_result.path.lstrip('/')

        s3_client_args = {}
        signature_version = config('yatai_service').get('S3_SIGNATURE_VERSION')
        s3_client_args['config'] = boto3.session.Config(
            signature_version=signature_version
        )
        if s3_endpoint_url is not None:
            s3_client_args['endpoint_url'] = s3_endpoint_url
        self.s3_client = boto3.client("s3", **s3_client_args)

    @property
    def _expiration(self):
        return config('yatai').getint('bento_uri_default_expiration')

    def _get_object_name(self, bento_name, bento_version):
        if self.base_path:
            return "/".join([self.base_path, bento_name, bento_version]) + '.tar.gz'
        else:
            return "/".join([bento_name, bento_version]) + '.tar.gz'

    def add(self, bento_name, bento_version):
        # Generate pre-signed s3 path for upload

        object_name = self._get_object_name(bento_name, bento_version)
        try:
            response = self.s3_client.generate_presigned_url(
                'put_object',
                Params={'Bucket': self.bucket, 'Key': object_name},
                ExpiresIn=self._expiration,
            )
        except Exception as e:
            raise YataiRepositoryException(
                "Not able to get pre-signed URL on S3. Error: {}".format(e)
            )

        return BentoUri(
            type=self.uri_type,
            uri='s3://{}/{}'.format(self.bucket, object_name),
            s3_presigned_url=response,
        )

    def get(self, bento_name, bento_version):
        # Return s3 path containing uploaded Bento files

        object_name = self._get_object_name(bento_name, bento_version)

        try:
            response = self.s3_client.generate_presigned_url(
                'get_object',
                Params={'Bucket': self.bucket, 'Key': object_name},
                ExpiresIn=self._expiration,
            )
            return response
        except Exception:  # pylint: disable=broad-except
            logger.error(
                "Failed generating presigned URL for downloading saved bundle from s3,"
                "falling back to using s3 path and client side credential for"
                "downloading with boto3"
            )
            return 's3://{}/{}'.format(self.bucket, object_name)

    def dangerously_delete(self, bento_name, bento_version):
        # Remove s3 path containing related Bento files

        object_name = self._get_object_name(bento_name, bento_version)

        try:
            response = self.s3_client.delete_object(Bucket=self.bucket, Key=object_name)
            DELETE_MARKER = 'DeleteMarker'  # whether object is successfully deleted.

            # Note: as of boto3 v1.13.13. delete_object returns an incorrect format as
            # expected from documentation.
            # Expected format:
            # {
            #   'DeleteMarker': True|False,
            #   'VersionId': 'string',
            #   'RequestCharged': 'requester'
            # }
            # Current return:
            # {
            #   'ResponseMetadata': {
            #     'RequestId': '****************',
            #     'HostId': '*****/******',
            #     'HTTPStatusCode': 204,
            #     'HTTPHeaders': {
            #       'x-amz-id-2': '*****/xxxxx',
            #       'x-amz-request-id': '332EE9F7AB555D2B',
            #        'date': 'Tue, 19 May 2020 19:46:57 GMT',
            #        'server': 'AmazonS3'
            #     },
            #     'RetryAttempts': 0
            #   }
            # }
            # An open issue on github: https://github.com/boto/boto3/issues/759
            if DELETE_MARKER in response:
                if response[DELETE_MARKER]:
                    return
                else:
                    logger.warning(
                        f"BentoML has deleted service '{bento_name}:{bento_version}' "
                        f"from YataiService records, but it failed to delete the saved "
                        f"bundle files stored in s3://{self.bucket}/{object_name}, "
                        f"the files may have already been deleted by the user."
                    )
                    return
            elif 'ResponseMetadata' in response:
                # Note: Use head_object to 'check' is the object deleted or not.
                # head_object only try to retrieve the metadata without returning
                # the object itself.
                try:
                    self.s3_client.head_object(Bucket=self.bucket, Key=object_name)
                    logger.warning(
                        f"BentoML has deleted service '{bento_name}:{bento_version}' "
                        f"from YataiService records, but it failed to delete the saved "
                        f"bundle files stored in s3://{self.bucket}/{object_name}, "
                        f"the files may have already been deleted by the user."
                    )
                except ClientError as e:
                    # expected ClientError with Code 404, as target object should be
                    # deleted and 'head_object' should raise
                    error_response = e.response.get('Error', {})
                    error_code = error_response.get('Code', None)
                    if error_code == '404':
                        # Error code 404 means target file object does not exist, as
                        # expected after delete_object call
                        return
                    else:
                        # unexpected boto3 ClientError
                        raise e
            else:
                raise YataiRepositoryException(
                    'Unrecognized response format from s3 delete_object'
                )
        except Exception as e:
            raise YataiRepositoryException(
                "Not able to delete object on S3. Error: {}".format(e)
            )


class BentoRepository(BentoRepositoryBase):
    def __init__(self, base_url=None, s3_endpoint_url=None):
        if base_url is None:
            base_url = config().get('default_repository_base_url')

        if is_s3_url(base_url):
            self._repo = _S3BentoRepository(base_url, s3_endpoint_url)
        else:
            self._repo = _LocalBentoRepository(base_url)

    def add(self, bento_name, bento_version):
        return self._repo.add(bento_name, bento_version)

    def get(self, bento_name, bento_version):
        return self._repo.get(bento_name, bento_version)

    def dangerously_delete(self, bento_name, bento_version):
        return self._repo.dangerously_delete(bento_name, bento_version)
