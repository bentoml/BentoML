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
from urllib.parse import urlparse

from simple_di import Provide, inject

from bentoml.configuration.containers import BentoMLContainer
from bentoml.exceptions import YataiRepositoryException
from bentoml.yatai.proto.repository_pb2 import BentoUri
from bentoml.yatai.repository.base_repository import BaseRepository

logger = logging.getLogger(__name__)


class GCSRepository(BaseRepository):
    @inject
    def __init__(
        self,
        base_url,
        expiration: int = Provide[
            BentoMLContainer.config.yatai.repository.gcs.expiration
        ],
    ):
        try:
            from google.cloud import storage
        except ImportError:
            raise YataiRepositoryException(
                '"google-cloud-storage" package is required for Google Cloud '
                'Storage Repository. You can install it with pip: '
                '"pip install google-cloud-storage"'
            )
        self.uri_type = BentoUri.GCS

        parse_result = urlparse(base_url)
        self.bucket = parse_result.netloc
        self.base_path = parse_result.path.lstrip('/')
        self.gcs_client = storage.Client()
        self.expiration = expiration

    def _get_object_name(self, bento_name, bento_version):
        if self.base_path:
            return "/".join([self.base_path, bento_name, bento_version]) + '.tar.gz'
        else:
            return "/".join([bento_name, bento_version]) + '.tar.gz'

    def add(self, bento_name, bento_version):
        object_name = self._get_object_name(bento_name, bento_version)
        try:
            bucket = self.gcs_client.bucket(self.bucket)
            blob = bucket.blob(object_name)

            response = blob.generate_signed_url(
                version="v4", expiration=self.expiration, method="PUT",
            )
        except Exception as e:
            raise YataiRepositoryException(
                "Not able to get pre-signed URL on GCS. Error: {}".format(e)
            )

        return BentoUri(
            type=self.uri_type,
            uri='gs://{}/{}'.format(self.bucket, object_name),
            gcs_presigned_url=response,
        )

    def get(self, bento_name, bento_version):
        # Return gcs path containing uploaded Bento files

        object_name = self._get_object_name(bento_name, bento_version)

        try:
            bucket = self.gcs_client.bucket(self.bucket)
            blob = bucket.blob(object_name)

            response = blob.generate_signed_url(
                version="v4", expiration=self.expiration, method="GET",
            )
            return response
        except Exception:  # pylint: disable=broad-except
            logger.error(
                "Failed generating presigned URL for downloading saved bundle from GCS,"
                "falling back to using gs path and client side credential for"
                "downloading with google.cloud.storage"
            )
            return 'gs://{}/{}'.format(self.bucket, object_name)

    def dangerously_delete(self, bento_name, bento_version):
        # Remove gcs path containing related Bento files

        object_name = self._get_object_name(bento_name, bento_version)

        try:
            bucket = self.gcs_client.bucket(self.bucket)
            blob = bucket.blob(object_name)
            blob.delete()
        except Exception as e:
            raise YataiRepositoryException(
                "Not able to delete object on GCS. Error: {}".format(e)
            )
