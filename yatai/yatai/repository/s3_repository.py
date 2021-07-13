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

import logging

from simple_di import Provide, inject

from bentoml.configuration.containers import BentoMLContainer
from bentoml.exceptions import YataiRepositoryException
from bentoml.yatai.proto.repository_pb2 import BentoUri
from bentoml.yatai.repository.base_repository import BaseRepository

logger = logging.getLogger(__name__)


class S3Repository(BaseRepository):
    @inject
    def __init__(
        self,
        base_url,
        endpoint_url: str = Provide[
            BentoMLContainer.config.yatai.repository.s3.endpoint_url
        ],
        signature_version: str = Provide[
            BentoMLContainer.config.yatai.repository.s3.signature_version
        ],
        expiration: int = Provide[
            BentoMLContainer.config.yatai.repository.s3.expiration
        ],
    ):
        import boto3
        from urllib.parse import urlparse

        self.uri_type = BentoUri.S3

        parse_result = urlparse(base_url)
        self.bucket = parse_result.netloc
        self.base_path = parse_result.path.lstrip('/')

        s3_client_args = {}
        s3_client_args['config'] = boto3.session.Config(
            signature_version=signature_version
        )
        if endpoint_url is not None:
            s3_client_args['endpoint_url'] = endpoint_url
        self.s3_client = boto3.client("s3", **s3_client_args)
        self.expiration = expiration

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
                ExpiresIn=self.expiration,
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
                ExpiresIn=self.expiration,
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

        from botocore.exceptions import ClientError

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
