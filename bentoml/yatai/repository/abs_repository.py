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
from datetime import datetime, timedelta
import os
from urllib.parse import urlparse
from simple_di import Provide, inject

from bentoml.configuration.containers import BentoMLContainer
from bentoml.exceptions import YataiRepositoryException
from bentoml.yatai.proto.repository_pb2 import BentoUri
from bentoml.yatai.repository.base_repository import BaseRepository

logger = logging.getLogger(__name__)


class ABSRepository(BaseRepository):
    @inject
    def __init__(
        self,
        base_url,
        expiration: int = Provide[
            BentoMLContainer.config.yatai.repository.abs.expiration
        ],
    ):
        try:
            from azure.storage.blob import (
                BlobServiceClient,
                generate_blob_sas,
                AccountSasPermissions,
            )
        except ImportError:
            raise YataiRepositoryException(
                '"azure-storage-blob" package is required for Azure Blob Storage.'
                'You can install it with pip: '
                '"pip install pip install azure-storage-blob"'
                'Find out more at https://docs.microsoft.com/en-us/azure/storage/blobs/storage-quickstart-blobs-python'  # noqa: E501
            )
        self.generate_blob_sas = generate_blob_sas
        # fmt: off
        self.sas_permission_add = AccountSasPermissions(
            create=True,
            write=True,
        )
        self.sas_permission_get = AccountSasPermissions(
            read=True,
        )
        # fmt: on

        self.uri_type = BentoUri.ABS
        self.base_url = base_url

        parsed_url = urlparse(base_url)
        self.account = parsed_url.netloc.split(".", 1)[0]
        self.container, self.blob = parsed_url.path.split("/", 1)

        connection_string = os.getenv("AZURE_STORAGE_CONNECTION_STRING")
        if not connection_string:
            raise YataiRepositoryException(
                "Not able to get AZURE_STORAGE_CONNECTION_STRING"
            )

        self.azure_blob_service_client = BlobServiceClient.from_connection_string(
            connection_string
        )
        self.expiration = expiration

    def _get_object_name(self, bento_name, bento_version):
        if self.blob:
            return "/".join([self.blob, bento_name, bento_version]) + '.tar.gz'
        else:
            return "/".join([bento_name, bento_version]) + '.tar.gz'

    def add(self, bento_name, bento_version):
        object_name = self._get_object_name(bento_name, bento_version)
        url = f"https://{self.account}.blob.core.windows.net/{self.container}/{object_name}"  # noqa: E501
        try:
            abs_container_client = self.azure_blob_service_client.get_container_client(
                self.container
            )
            abs_blob_client = abs_container_client.get_blob_client(object_name)
            sas_token = self.generate_blob_sas(
                account_name=self.account,
                account_key=abs_blob_client.credential.account_key,
                container_name=self.container,
                blob_name=object_name,
                permission=self.sas_permission_add,
                expiry=datetime.utcnow() + timedelta(0, self.expiration),
            )
            sas_url = f"{url}?{sas_token}"
        except Exception as e:
            raise YataiRepositoryException(
                "Not able to get pre-signed URL on ABS. Error: {}".format(e)
            )

        # fmt: off
        return BentoUri(
            type=self.uri_type,
            uri=url,
            abs_presigned_url=sas_url,
        )
        # fmt: on

    def get(self, bento_name, bento_version):
        # Return abs path containing uploaded Bento files

        object_name = self._get_object_name(bento_name, bento_version)
        url = f"https://{self.account}.blob.core.windows.net/{self.container}/{object_name}"  # noqa: E501
        try:
            abs_container_client = self.azure_blob_service_client.get_container_client(
                self.container
            )
            abs_blob_client = abs_container_client.get_blob_client(object_name)
            sas_token = self.generate_blob_sas(
                account_name=self.account,
                account_key=abs_blob_client.credential.account_key,
                container_name=self.container,
                blob_name=object_name,
                permission=self.sas_permission_get,
                expiry=datetime.utcnow() + timedelta(0, self.expiration),
            )
            return f"{url}?{sas_token}"
        except Exception:  # pylint: disable=broad-except
            logger.error(
                "Failed generating presigned URL for downloading saved bundle from ABS,"
                "falling back to using azure path and client side credential for"
                "downloading with azure.blob.storage"
            )
            return url

    def dangerously_delete(self, bento_name, bento_version):
        # Remove abs path containing related Bento files

        object_name = self._get_object_name(bento_name, bento_version)
        try:
            abs_container_client = self.azure_blob_service_client.get_container_client(
                self.container
            )
            abs_blob_client = abs_container_client.get_blob_client(object_name)
            abs_blob_client.delete_blob()
        except Exception as e:
            raise YataiRepositoryException(
                "Not able to delete object on ABS. Error: {}".format(e)
            )
