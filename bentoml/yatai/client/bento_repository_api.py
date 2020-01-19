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

# List of APIs for accessing remote or local yatai service via Python

import io
import os
import json
import logging
import tarfile
import requests
import shutil


from bentoml.exceptions import BentoMLException
from bentoml.proto.repository_pb2 import (
    AddBentoRequest,
    GetBentoRequest,
    BentoUri,
    UpdateBentoRequest,
    UploadStatus,
    ListBentoRequest,
    DangerouslyDeleteBentoRequest,
)
from bentoml.proto import status_pb2
from bentoml.utils.tempdir import TempDirectory
from bentoml.bundler import save_to_dir, load_bento_service_metadata
from bentoml.yatai.status import Status


logger = logging.getLogger(__name__)


class BentoRepositoryAPIClient:
    def __init__(self, yatai_service):
        self.yatai_service = yatai_service

    def upload(self, bento_service, version=None):
        """Save and upload given bento_service to yatai_service, which manages all your
        saved BentoService bundles and model serving deployments.

        Args:
            bento_service (bentoml.service.BentoService): a Bento Service instance
            version (str): optional,
        Return:
            URI to where the BentoService is being saved to
        """
        with TempDirectory() as tmpdir:
            save_to_dir(bento_service, tmpdir, version)
            return self._upload_bento_service(tmpdir)

    def _upload_bento_service(self, saved_bento_path):
        bento_service_metadata = load_bento_service_metadata(saved_bento_path)

        get_bento_response = self.yatai_service.GetBento(
            GetBentoRequest(
                bento_name=bento_service_metadata.name,
                bento_version=bento_service_metadata.version,
            )
        )
        if get_bento_response.status.status_code == status_pb2.Status.OK:
            raise BentoMLException(
                "BentoService bundle {}:{} already registered in repository. Reset "
                "BentoService version with BentoService#set_version or bypass BentoML's"
                " model registry feature with BentoService#save_to_dir".format(
                    bento_service_metadata.name, bento_service_metadata.version
                )
            )
        elif get_bento_response.status.status_code != status_pb2.Status.NOT_FOUND:
            raise BentoMLException(
                'Failed accessing YataiService. {error_code}:'
                '{error_message}'.format(
                    error_code=Status.Name(get_bento_response.status.status_code),
                    error_message=get_bento_response.status.error_message,
                )
            )
        request = AddBentoRequest(
            bento_name=bento_service_metadata.name,
            bento_version=bento_service_metadata.version,
        )
        response = self.yatai_service.AddBento(request)

        if response.status.status_code != status_pb2.Status.OK:
            raise BentoMLException(
                "Error adding BentoService bundle to repository: {}:{}".format(
                    Status.Name(response.status.status_code),
                    response.status.error_message,
                )
            )

        if response.uri.type == BentoUri.LOCAL:
            if os.path.exists(response.uri.uri):
                # due to copytree dst must not already exist
                shutil.rmtree(response.uri.uri)
            shutil.copytree(saved_bento_path, response.uri.uri)

            self._update_bento_upload_progress(bento_service_metadata)

            logger.info(
                "BentoService bundle '%s:%s' created at: %s",
                bento_service_metadata.name,
                bento_service_metadata.version,
                response.uri.uri,
            )
            # Return URI to saved bento in repository storage
            return response.uri.uri
        elif response.uri.type == BentoUri.S3:
            self._update_bento_upload_progress(
                bento_service_metadata, UploadStatus.UPLOADING, 0
            )

            fileobj = io.BytesIO()
            with tarfile.open(mode="w:gz", fileobj=fileobj) as tar:
                tar.add(saved_bento_path, arcname=bento_service_metadata.name)
            fileobj.seek(0, 0)

            files = {'file': ('dummy', fileobj)}  # dummy file name because file name
            # has been generated when getting the pre-signed signature.
            data = json.loads(response.uri.additional_fields)
            uri = data.pop('url')
            http_response = requests.post(uri, data=data, files=files)

            if http_response.status_code != 204:
                self._update_bento_upload_progress(
                    bento_service_metadata, UploadStatus.ERROR
                )

                raise BentoMLException(
                    "Error saving BentoService bundle to S3. {}: {} ".format(
                        Status.Name(http_response.status_code), http_response.text
                    )
                )

            self._update_bento_upload_progress(bento_service_metadata)

            logger.info(
                "Successfully saved BentoService bundle '%s:%s' to S3: %s",
                bento_service_metadata.name,
                bento_service_metadata.version,
                response.uri.uri,
            )

            return response.uri.uri

        else:
            raise BentoMLException(
                f"Error saving Bento to target repository, URI type {response.uri.type}"
                f" at {response.uri.uri} not supported"
            )

    def _update_bento_upload_progress(
        self, bento_service_metadata, status=UploadStatus.DONE, percentage=None
    ):
        upload_status = UploadStatus(status=status, percentage=percentage)
        upload_status.updated_at.GetCurrentTime()
        update_bento_req = UpdateBentoRequest(
            bento_name=bento_service_metadata.name,
            bento_version=bento_service_metadata.version,
            upload_status=upload_status,
            service_metadata=bento_service_metadata,
        )
        self.yatai_service.UpdateBento(update_bento_req)

    def get(self, bento_name, bento_version=None):
        get_bento_request = GetBentoRequest(
            bento_name=bento_name, bento_version=bento_version
        )
        return self.yatai_service.GetBento(get_bento_request)

    def list(
        self,
        bento_name=None,
        offset=None,
        limit=None,
        order_by=None,
        ascending_order=None,
    ):
        list_bento_request = ListBentoRequest(
            bento_name=bento_name,
            offset=offset,
            limit=limit,
            order_by=order_by,
            ascending_order=ascending_order,
        )
        return self.yatai_service.ListBento(list_bento_request)

    def dangerously_delete_bento(self, name, version):
        dangerously_delete_bento_request = DangerouslyDeleteBentoRequest(
            bento_name=name, bento_version=version
        )
        return self.yatai_service.DangerouslyDeleteBento(
            dangerously_delete_bento_request
        )
