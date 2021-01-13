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
import logging
import tarfile
import requests
import shutil


from bentoml.exceptions import BentoMLException
from bentoml.utils import status_pb_to_error_code_and_message
from bentoml.utils.lazy_loader import LazyLoader
from bentoml.utils.usage_stats import track
from bentoml.yatai.client.label_utils import generate_gprc_labels_selector
from bentoml.yatai.label_store import _validate_labels
from bentoml.yatai.proto.repository_pb2 import (
    AddBentoRequest,
    GetBentoRequest,
    BentoUri,
    UpdateBentoRequest,
    UploadStatus,
    ListBentoRequest,
    DangerouslyDeleteBentoRequest,
    ContainerizeBentoRequest,
)
from bentoml.yatai.proto import status_pb2
from bentoml.utils.tempdir import TempDirectory
from bentoml.saved_bundle import save_to_dir, load_bento_service_metadata, safe_retrieve
from bentoml.yatai.status import Status


logger = logging.getLogger(__name__)
yatai_proto = LazyLoader('yatai_proto', globals(), 'bentoml.yatai.proto')


class BentoRepositoryAPIClient:
    def __init__(self, yatai_service):
        # YataiService stub for accessing remote YataiService RPCs
        self.yatai_service = yatai_service

    def push(self, bento, labels=None):
        """
        Push a local BentoService to a remote yatai server.

        Args:
            bento: a BentoService identifier in the format of NAME:VERSION
            labels: optional. List of labels for the BentoService.

        Returns:
            BentoService saved path

        Example:

        >>> svc = MyBentoService()
        >>> svc.save()
        >>>
        >>> remote_yatai_client = get_yatai_client('http://remote.yatai.service:50050')
        >>> bento = f'{svc.name}:{svc.version}'
        >>> remote_saved_path= remote_yatai_client.repository.push(bento)
        """
        track('py-api-push')

        from bentoml.yatai.client import get_yatai_client

        local_yc = get_yatai_client()

        local_bento_pb = local_yc.repository.get(bento)
        if local_bento_pb.uri.s3_presigned_url:
            bento_bundle_path = local_bento_pb.uri.s3_presigned_url
        elif local_bento_pb.uri.gcs_presigned_url:
            bento_bundle_path = local_bento_pb.uri.gcs_presigned_url
        else:
            bento_bundle_path = local_bento_pb.uri.uri
        return self.upload_from_dir(bento_bundle_path, labels=labels)

    def pull(self, bento):
        """
        Pull a BentoService from a remote yatai service. The BentoService will be saved
        and registered with local yatai service.

        Args:
            bento: a BentoService identifier in the form of NAME:VERSION

        Returns:
            BentoService saved path

        Example:

        >>> client = get_yatai_client('127.0.0.1:50051')
        >>> saved_path = client.repository.pull('MyService:')
        """
        track('py-api-pull')

        bento_pb = self.get(bento)
        with TempDirectory() as tmpdir:
            # Create a non-exist directory for safe_retrieve
            target_bundle_path = os.path.join(tmpdir, 'bundle')
            self.download_to_directory(bento_pb, target_bundle_path)

            from bentoml.yatai.client import get_yatai_client

            local_yc = get_yatai_client()
            return local_yc.repository.upload_from_dir(target_bundle_path)

    def upload(self, bento_service, version=None, labels=None):
        """Save and upload given bento_service to yatai_service, which manages all your
        saved BentoService bundles and model serving deployments.

        Args:
            bento_service (bentoml.service.BentoService): a Bento Service instance
            version (str): optional,
            labels (dict): optional
        Return:
            URI to where the BentoService is being saved to
        """
        with TempDirectory() as tmpdir:
            save_to_dir(bento_service, tmpdir, version, silent=True)
            return self.upload_from_dir(tmpdir, labels)

    def upload_from_dir(self, saved_bento_path, labels=None):
        bento_service_metadata = load_bento_service_metadata(saved_bento_path)
        if labels:
            _validate_labels(labels)
            bento_service_metadata.labels.update(labels)

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
                "BentoService bundle '%s:%s' saved to: %s",
                bento_service_metadata.name,
                bento_service_metadata.version,
                response.uri.uri,
            )
            # Return URI to saved bento in repository storage
            return response.uri.uri
        elif response.uri.type == BentoUri.S3 or response.uri.type == BentoUri.GCS:
            uri_type = 'S3' if response.uri.type == BentoUri.S3 else 'GCS'
            self._update_bento_upload_progress(
                bento_service_metadata, UploadStatus.UPLOADING, 0
            )

            fileobj = io.BytesIO()
            with tarfile.open(mode="w:gz", fileobj=fileobj) as tar:
                tar.add(saved_bento_path, arcname=bento_service_metadata.name)
            fileobj.seek(0, 0)

            if response.uri.type == BentoUri.S3:
                http_response = requests.put(
                    response.uri.s3_presigned_url, data=fileobj
                )
            elif response.uri.type == BentoUri.GCS:
                http_response = requests.put(
                    response.uri.gcs_presigned_url, data=fileobj
                )

            if http_response.status_code != 200:
                self._update_bento_upload_progress(
                    bento_service_metadata, UploadStatus.ERROR
                )
                raise BentoMLException(
                    f"Error saving BentoService bundle to {uri_type}."
                    f"{http_response.status_code}: {http_response.text}"
                )

            self._update_bento_upload_progress(bento_service_metadata)

            logger.info(
                "Successfully saved BentoService bundle '%s:%s' to {uri_type}: %s",
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

    def download_to_directory(self, bento_pb, target_dir):
        if bento_pb.uri.s3_presigned_url:
            bento_service_bundle_path = bento_pb.uri.s3_presigned_url
        elif bento_pb.uri.gcs_presigned_url:
            bento_service_bundle_path = bento_pb.uri.gcs_presigned_url
        else:
            bento_service_bundle_path = bento_pb.uri.uri

        safe_retrieve(bento_service_bundle_path, target_dir)

    def get(self, bento):
        """
        Get a BentoService info

        Args:
            bento: a BentoService identifier in the format of NAME:VERSION

        Returns:
            bentoml.yatai.proto.repository_pb2.Bento

        Example:

        >>> yatai_client = get_yatai_client()
        >>> bento_info = yatai_client.repository.get('my_service:version')
        """
        track('py-api-get')
        if ':' not in bento:
            raise BentoMLException(
                'BentoService name or version is missing. Please provide in the '
                'format of name:version'
            )
        name, version = bento.split(':')
        result = self.yatai_service.GetBento(
            GetBentoRequest(bento_name=name, bento_version=version)
        )
        if result.status.status_code != yatai_proto.status_pb2.Status.OK:
            error_code, error_message = status_pb_to_error_code_and_message(
                result.status
            )
            raise BentoMLException(
                f'BentoService {name}:{version} not found - '
                f'{error_code}:{error_message}'
            )
        return result.bento

    def list(
        self,
        bento_name=None,
        offset=None,
        limit=None,
        labels=None,
        order_by=None,
        ascending_order=None,
    ):
        """
        List BentoServices that satisfy the specified criteria.

        Args:
            bento_name: optional. BentoService name
            limit: optional. maximum number of returned results
            labels: optional.
            offset: optional. offset of results
            order_by: optional. order by results
            ascending_order:  optional. direction of results order

        Returns:
            [bentoml.yatai.proto.repository_pb2.Bento]

        Example:

        >>> yatai_client = get_yatai_client()
        >>> bentos_info_list = yatai_client.repository.list(
        >>>     labels='key=value,key2=value'
        >>> )
        """
        track('py-api-list')
        list_bento_request = ListBentoRequest(
            bento_name=bento_name,
            offset=offset,
            limit=limit,
            order_by=order_by,
            ascending_order=ascending_order,
        )
        if labels is not None:
            generate_gprc_labels_selector(list_bento_request.label_selectors, labels)

        result = self.yatai_service.ListBento(list_bento_request)
        if result.status.status_code != yatai_proto.status_pb2.Status.OK:
            error_code, error_message = status_pb_to_error_code_and_message(
                result.status
            )
            raise BentoMLException(f'{error_code}:{error_message}')
        return result.bentos

    def delete(self, bento):
        """
        Delete bento

        Args:
            bento: a BentoService identifier in the format of NAME:VERSION

        Example:
        >>>
        >>> yatai_client = get_yatai_client()
        >>> yatai_client.repository.delete('my_service:version')
        """
        track('py-api-delete')
        if ':' not in bento:
            raise BentoMLException(
                'BentoService name or version is missing. Please provide in the '
                'format of name:version'
            )
        name, version = bento.split(':')
        result = self.yatai_service.DangerouslyDeleteBento(
            DangerouslyDeleteBentoRequest(bento_name=name, bento_version=version)
        )
        if result.status.status_code != yatai_proto.status_pb2.Status.OK:
            error_code, error_message = status_pb_to_error_code_and_message(
                result.status
            )
            raise BentoMLException(
                f'Failed to delete Bento {bento} {error_code}:{error_message}'
            )

    def prune(self, bento_name=None, labels=None):
        """
        Delete all BentoServices that matches the specified criteria

        Args:
            bento_name: optional
            labels: optional

        Example:

        >>> yatai_client = get_yatai_client()
        >>> # Delete all bento services
        >>> yatai_client.repository.prune()
        >>> # Delete bento services that matches with the label `ci=failed`
        >>> yatai_client.repository.prune(labels='ci=failed')
        """
        track('py-api-prune')
        list_bentos_result = self.list(bento_name=bento_name, labels=labels,)
        if list_bentos_result.status.status_code != yatai_proto.status_pb2.Status.OK:
            error_code, error_message = status_pb_to_error_code_and_message(
                list_bentos_result.status
            )
            raise BentoMLException(f'{error_code}:{error_message}')
        for bento in list_bentos_result.bentos:
            bento_tag = f'{bento.name}:{bento.version}'
            try:
                self.delete(bento_tag)
            except BentoMLException as e:
                logger.error(f'Failed to delete Bento {bento_tag}: {e}')

    def containerize(self, bento, tag=None, build_args=None, push=False):
        """
        Create a container image from a BentoService.

        Args:
            bento: string
            tag: string
            build_args: dict
            push: boolean

        Returns:
            Image tag: String
        """
        track('py-api-containerize')
        if ':' not in bento:
            raise BentoMLException(
                'BentoService name or version is missing. Please provide in the '
                'format of name:version'
            )
        name, version = bento.split(':')
        containerize_request = ContainerizeBentoRequest(
            bento_name=name,
            bento_version=version,
            tag=tag,
            build_args=build_args,
            push=push,
        )
        result = self.yatai_service.ContainerizeBento(containerize_request)

        if result.status.status_code != yatai_proto.status_pb2.Status.OK:
            error_code, error_message = status_pb_to_error_code_and_message(
                result.status
            )
            raise BentoMLException(
                f'Failed to containerize {bento} - {error_code}:{error_message}'
            )
        return result.tag
