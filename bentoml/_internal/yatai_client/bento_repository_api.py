import io
import logging
import os
import shutil
import tarfile
import uuid
from typing import TYPE_CHECKING, Dict, List, Optional

import click
import grpc
import requests

from bentoml.exceptions import BentoMLException, BentoMLRpcError

# from ..bundle import (
#     load_bento_service_metadata,
#     load_from_dir,
#     safe_retrieve,
#     save_to_dir,
# )
from ..utils import (
    is_gcs_url,
    is_s3_url,
    resolve_bento_bundle_uri,
    status_pb_to_error_code_and_message,
)
from ..utils.lazy_loader import LazyLoader
from ..utils.tempdir import TempDirectory
from .label_utils import generate_gprc_labels_selector

# from .yatai.grpc_stream_utils import UploadBentoStreamRequests
from .proto import status_pb2
from .proto.repository_pb2 import (
    AddBentoRequest,
    BentoUri,
    ContainerizeBentoRequest,
    DangerouslyDeleteBentoRequest,
    DownloadBentoRequest,
    GetBentoRequest,
    ListBentoRequest,
    UpdateBentoRequest,
    UploadStatus,
)
from .proto.yatai_service_pb2_grpc import YataiStub

# from .status import Status

if TYPE_CHECKING:
    from ..service import Service
    from . import YataiClient
    from .proto.repository_pb2 import Bento

logger = logging.getLogger(__name__)
yatai_proto = LazyLoader("yatai_proto", globals(), "bentoml.yatai.proto")

# Default timeout in seconds per grpc request.
DEFAULT_GRPC_REQUEST_TIMEOUT = 6


def is_remote_yatai(yatai_service) -> bool:
    return isinstance(yatai_service, YataiStub)


class BentoRepositoryAPIClient:
    """
    Provided API to manage :class:`~bentoml.Service`
    for local Yatai repository
    """

    def __init__(self, yatai_service):
        # YataiService stub for accessing remote YataiService RPCs
        self.yatai_service: "YataiStub" = yatai_service

    def push(self, bento: str, with_labels: bool = True) -> "BentoUri":
        """
        Push a local Service to a remote yatai server.

        Args:
            bento (`str`):
                A Service identifier in the format of ``NAME:VERSION``.

            with_labels (`bool`, `optional`):
                Whether to push Service with given label.

        Returns: URI as gRPC stub for Service saved location.

        Example::

            from bentoml.yatai.client import get_yatai_client
            svc = MyService()
            svc.save()

            remote_yatai_client = get_yatai_client('http://remote.yatai.service:50050')
            bento_name = f'{svc.name}:{svc.version}'
            remote_saved_path= remote_yatai_client.repository.push(bento_name)
        """
        from bentoml.yatai.client import get_yatai_client

        local_yc: "YataiClient" = get_yatai_client()

        local_bento_pb: "Bento" = local_yc.repository.get(bento)
        if local_bento_pb.uri.s3_presigned_url:
            bento_bundle_path = local_bento_pb.uri.s3_presigned_url
        elif local_bento_pb.uri.gcs_presigned_url:
            bento_bundle_path = local_bento_pb.uri.gcs_presigned_url
        else:
            bento_bundle_path = local_bento_pb.uri.uri
        labels = (
            dict(local_bento_pb.bento_service_metadata.labels)
            if with_labels is True and local_bento_pb.bento_service_metadata.labels
            else None
        )

        return self.upload_from_dir(bento_bundle_path, labels=labels)

    def pull(self, bento: str) -> "BentoUri":
        """
        Pull a :class:`~bentoml.Service` from remote Yatai.
        The Service will then be saved and registered to local Yatai.

        Args:
            bento (`str`):
                a Service identifier in the form of ``NAME:VERSION``

        Returns:
            :class:`reflection.GeneratedProtocolMessageType`:
                URI as gRPC stub for save location of Service.

        Example::

            from bentoml.yatai.client import get_yatai_client
            client = get_yatai_client('127.0.0.1:50051')
            saved_path = client.repository.pull('MyService:20210808_E38F3')
        """
        bento_pb: "Bento" = self.get(bento)
        with TempDirectory() as tmpdir:
            # Create a non-exist directory for safe_retrieve
            target_bundle_path = os.path.join(tmpdir, "bundle")
            self.download_to_directory(bento_pb, target_bundle_path)

            from bentoml.yatai.client import get_yatai_client

            labels = (
                dict(bento_pb.bento_service_metadata.labels)
                if bento_pb.bento_service_metadata.labels
                else None
            )

            local_yc = get_yatai_client()
            return local_yc.repository.upload_from_dir(
                target_bundle_path, labels=labels
            )

    def upload(
        self, bento_service: "Service", version: str = None, labels: Dict = None,
    ) -> "BentoUri":
        """
        Save and upload given :class:`~bentoml.Service`
        to YataiService.

        Args:
            bento_service (:class:`~bentoml.Service`):
                a Service instance
            version (`str`, `optional`):
                version of ``bento_service``
            labels (`dict`, `optional`):
                :class:`~bentoml.Service` metadata

        Returns:
            BentoUri as gRPC stub for save location of Service.

        Example::

            from bentoml.yatai.client import get_yatai_client

            svc = MyService()
            svc.save()

            remote_yatai_client = get_yatai_client('https://remote.yatai.service:50050')
            remote_path = remote_yatai_client.repository.upload(svc)
        """
        with TempDirectory() as tmpdir:
            save_to_dir(bento_service, tmpdir, version, silent=True)
            return self.upload_from_dir(tmpdir, labels)

    def upload_from_dir(self, saved_bento_path: str, labels: Dict = None) -> "BentoUri":
        from bentoml.yatai.db.stores.label import _validate_labels

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
                "Service bundle {}:{} already registered in repository. Reset "
                "Service version with Service#set_version or bypass BentoML's"
                " model registry feature with Service#save_to_dir".format(
                    bento_service_metadata.name, bento_service_metadata.version
                )
            )
        elif get_bento_response.status.status_code != status_pb2.Status.NOT_FOUND:
            raise BentoMLException(
                "Failed accessing YataiService. {error_code}:"
                "{error_message}".format(
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
                "Error adding Service bundle to repository: {}:{}".format(
                    Status.Name(response.status.status_code),
                    response.status.error_message,
                )
            )

        if response.uri.type == BentoUri.LOCAL:
            # When using Yatai backed by File System repository,
            # if Yatai is a local instance, copy the files directly.
            # Otherwise, use UploadBento RPC to stream files to remote Yatai server
            if is_remote_yatai(self.yatai_service):
                self._upload_bento(
                    bento_service_metadata.name,
                    bento_service_metadata.version,
                    saved_bento_path,
                )
                update_bento_service = UpdateBentoRequest(
                    bento_name=bento_service_metadata.name,
                    bento_version=bento_service_metadata.version,
                    service_metadata=bento_service_metadata,
                )
                self.yatai_service.UpdateBento(update_bento_service)
            else:
                if os.path.exists(response.uri.uri):
                    raise BentoMLException(
                        f"Bento bundle directory {response.uri.uri} already exist"
                    )
                shutil.copytree(saved_bento_path, response.uri.uri)
                upload_status = UploadStatus.DONE

                self._update_bento_upload_progress(
                    bento_service_metadata, status=upload_status
                )

            logger.info(
                "Service bundle '%s:%s' saved to: %s",
                bento_service_metadata.name,
                bento_service_metadata.version,
                response.uri.uri,
            )
            # Return URI to saved bento in repository storage
            return response.uri.uri
        elif response.uri.type == BentoUri.S3 or response.uri.type == BentoUri.GCS:
            uri_type = "S3" if response.uri.type == BentoUri.S3 else "GCS"
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
                    f"Error saving Service bundle to {uri_type}."
                    f"{http_response.status_code}: {http_response.text}"
                )

            self._update_bento_upload_progress(bento_service_metadata)

            logger.info(
                "Successfully saved Service bundle '%s:%s' to {uri_type}: %s",
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

    def download_to_directory(self, bento_pb, target_dir: str) -> None:
        """
        Download or move bundle bundle to target directory.

        Args:
            bento_pb: bento bundle protobuf dict
            target_dir (`str`):

        Returns:
            None

        Raises:
            BentoMLException:
                Unrecognised Bento bundle storage type
        """
        if bento_pb.uri.type == BentoUri.S3:
            bento_service_bundle_path = bento_pb.uri.s3_presigned_url
        elif bento_pb.uri.type == BentoUri.GCS:
            bento_service_bundle_path = bento_pb.uri.gcs_presigned_url
        elif bento_pb.uri.type == BentoUri.LOCAL:
            # Download from remote yatai otherwise provide the file path.
            if is_remote_yatai(self.yatai_service):
                bento_service_bundle_path = self._download_bento(
                    bento_pb.name, bento_pb.version
                )
            else:
                bento_service_bundle_path = bento_pb.uri.uri
        else:
            raise BentoMLException(
                f"Unrecognized Bento bundle storage type {bento_pb.uri.type}"
            )

        safe_retrieve(bento_service_bundle_path, target_dir)

    def get(self, bento: str) -> "Bento":
        """
        Args:
            bento (`str`):
                A Service identifier in the format of ``NAME:VERSION``

        Returns:
            :class:`~bentoml.Service` metadata from Yatai RPC server.

        Raises:
            BentoMLException: ``bento`` is missing or have invalid format.

        Example::

            from bentoml.yatai.client import get_yatai_client
            yatai_client = get_yatai_client()
            bento_info = yatai_client.repository.get('my_service:version')
        """
        if ":" not in bento:
            raise BentoMLException(
                "Service name or version is missing. Please provide in the "
                "format of name:version"
            )
        name, version = bento.split(":")
        result = self.yatai_service.GetBento(
            GetBentoRequest(bento_name=name, bento_version=version)
        )
        if result.status.status_code != yatai_proto.status_pb2.Status.OK:
            error_code, error_message = status_pb_to_error_code_and_message(
                result.status
            )
            raise BentoMLException(
                f"Service {name}:{version} not found - " f"{error_code}:{error_message}"
            )
        return result.bento

    def list(
        self,
        bento_name: str = None,
        offset: int = None,
        limit: int = None,
        order_by: str = None,
        ascending_order: bool = None,
        labels: str = None,
    ) -> List["Bento"]:
        """
        List Services that satisfy the specified criteria.

        Args:
            bento_name (`str`):
                Service name
            offset (`int`):
                offset of results
            limit (`int`):
                maximum number of returned results
            labels (`str`):
                sorted by given labels
            order_by (`str`):
                orders retrieved Service by :obj:`created_at` or :obj:`name`
            ascending_order (`bool`):
                direction of results order

        Returns:
            lists of :class:`~bentoml.Service` metadata.

        Example::

            from bentoml.yatai.client import get_yatai_client
            yatai_client = get_yatai_client()
            bentos_info_list = yatai_client.repository.list(labels='key=value,key2=value')
        """  # noqa: E501

        # TODO: ignore type checking for this function. This is
        #  due to all given arguments in `ListBentoRequest` are
        #  not optional types. One solution is to make all
        #  `ListBentoRequest` args in `list` positional. This could
        #  introduce different behaviour at different places in the
        #  codebase. Low triage
        list_bento_request = ListBentoRequest(
            bento_name=bento_name,  # type: ignore
            offset=offset,  # type: ignore
            limit=limit,  # type: ignore
            order_by=order_by,  # type: ignore
            ascending_order=ascending_order,  # type: ignore
        )

        if labels is not None:
            generate_gprc_labels_selector(list_bento_request.label_selectors, labels)

        result = self.yatai_service.ListBento(list_bento_request)
        if result.status.status_code != yatai_proto.status_pb2.Status.OK:
            error_code, error_message = status_pb_to_error_code_and_message(
                result.status
            )
            raise BentoMLException(f"{error_code}:{error_message}")
        return result.bentos

    def _delete_bento_bundle(self, bento_tag, require_confirm):
        bento_pb = self.get(bento_tag)
        if require_confirm and not click.confirm(f"Permanently delete {bento_tag}?"):
            return
        result = self.yatai_service.DangerouslyDeleteBento(
            DangerouslyDeleteBentoRequest(
                bento_name=bento_pb.name, bento_version=bento_pb.version
            )
        )

        if result.status.status_code != yatai_proto.status_pb2.Status.OK:
            error_code, error_message = status_pb_to_error_code_and_message(
                result.status
            )
            # Rather than raise Exception, continue to delete the next bentos
            logger.error(
                f"Failed to delete {bento_pb.name}:{bento_pb.version} - "
                f"{error_code}:{error_message}"
            )
        else:
            logger.info(f"Deleted {bento_pb.name}:{bento_pb.version}")

    def delete(
        self,
        bento_tag: str = None,
        labels: str = None,
        bento_name: str = None,
        bento_version: str = None,
        prune: Optional[bool] = False,  # pylint: disable=redefined-builtin
        require_confirm: Optional[bool] = False,
    ) -> None:
        """
        Delete bentos that matches the specified criteria.

        Args:
            bento_tag (`str`, `optional`):
                Service tags
            labels (`str`, `optional`):
                :class:`~bentoml.Service` labels
            bento_name (`str`, `optional`):
                given name of Service
            bento_version (`str`, `optional`):
                versions of given Service
            prune (`bool`, `optional`):
                Whether to delete all Service, default to ``False``
            require_confirm (`bool`, `optional`):
                Requires to confirm delete, default to ``False``

        Raises:
            BentoMLException: ``bento_tag``, ``bento_name`` and ``bento_version`` are parsed
                                at the same time

        Example::

            from bentoml.yatai.client import get_yatai_client

            yatai_client = get_yatai_client()
            # Delete all bento services
            yatai_client.repository.delete(prune=True)
            # Delete bento service with name is `IrisClassifier` and version `0.1.0`
            yatai_client.repository.delete(
                bento_name='IrisClassifier', bento_version='0.1.0'
            )
            # or use bento tag
            yatai_client.repository.delete('IrisClassifier:v0.1.0')
            # Delete all bento services with name 'MyService`
            yatai_client.repository.delete(bento_name='MyService')
            # Delete all bento services with labels match `ci=failed` and `cohort=20`
            yatai_client.repository.delete(labels='ci=failed, cohort=20')
        """  # noqa: E501
        delete_list_limit = 50

        if (
            bento_tag is not None
            and bento_name is not None
            and bento_version is not None
        ):
            raise BentoMLException("Too much arguments")

        if bento_tag is not None:
            logger.info(f"Deleting saved Bento bundle {bento_tag}")
            return self._delete_bento_bundle(bento_tag, require_confirm)
        elif bento_name is not None and bento_tag is not None:
            logger.info(f"Deleting saved Bento bundle {bento_name}:{bento_version}")
            return self._delete_bento_bundle(
                f"{bento_name}:{bento_version}", require_confirm
            )
        else:
            # list of bentos
            if prune is True:
                logger.info("Deleting all BentoML saved bundles.")
                # ignore other fields
                bento_name = None
                labels = None
            else:
                log_message = "Deleting saved Bento bundles"
                if bento_name is not None:
                    log_message += f" with name: {bento_name},"
                if labels is not None:
                    log_message += f" with labels match to {labels}"
                logger.info(log_message)
            offset = 0
            while offset >= 0:
                bento_list = self.list(
                    bento_name=bento_name,
                    labels=labels,
                    offset=offset,
                    limit=delete_list_limit,
                )
                offset += delete_list_limit
                # Stop the loop, when no more bentos
                if len(bento_list) == 0:
                    break
                else:
                    for bento in bento_list:
                        self._delete_bento_bundle(
                            f"{bento.name}:{bento.version}", require_confirm
                        )

    def containerize(
        self,
        bento: str,
        tag: str,
        build_args: Dict[str, str] = None,
        push: bool = False,
    ) -> str:
        """
        Create a container image from a :class:`~bentoml.Service`

        Args:
            bento (`str`):
                A Service identifier with ``NAME:VERSION`` format.
            tag (`str`):
                Service tag.
            build_args (`Dict[str, str]`, `optional`):
                Build args to parse to ``docker build``
            push (`bool`, `optional`):
                Whether to push built container to remote YataiService.

        Returns:
            :obj:`str` representing the image tag of the containerized :class:`~bentoml.Service`

        Raises:
            BentoMLException: ``bento`` is missing or incorrect format.

        Example::

            svc = MyModelService()
            svc.save()

            from bentoml.yatai.client import get_yatai_client

            yatai_client = get_yatai_client()
            tag = yatai_client.repository.containerize(f'{svc.name}:{svc.version}')
        """  # noqa: E501
        if ":" not in bento:
            raise BentoMLException(
                "Service name or version is missing. Please provide in the "
                "format of name:version"
            )
        name, version = bento.split(":")
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
                f"Failed to containerize {bento} - {error_code}:{error_message}"
            )
        return result.tag

    def load(self, bento: str) -> "Service":
        """
        Load :class:`~bentoml.Service` from nametag identifier
        or from a bento bundle path.

        Args:
            bento (`str`):
                :class:`~bentoml.Service` identifier or bundle path. Note
                that nametag identifier will have the following format: :code:`NAME:VERSION`

        Returns:
            :class:`~bentoml.Service`

        Example::

            from bentoml.yatai.client import get_yatai_client
            yatai_client = get_yatai_client()

            # Load Service bases on bento tag.
            bento_from_name = yatai_client.repository.load('service_name:version')
            # Load Service from bento bundle path
            bento_from_path = yatai_client.repository.load('/path/to/bento/bundle')
            # Load Service from s3 storage
            bento_from_reg = yatai_client.repository.load('s3://bucket/path/bundle.tar.gz')
        """  # noqa: E501
        if os.path.isdir(bento) or is_s3_url(bento) or is_gcs_url(bento):
            saved_bundle_path = bento
        else:
            bento_pb = self.get(bento)
            if bento_pb.uri.type == BentoUri.LOCAL and is_remote_yatai(
                self.yatai_service
            ):
                saved_bundle_path = self._download_bento(
                    bento_pb.name, bento_pb.version
                )
            else:
                saved_bundle_path = resolve_bento_bundle_uri(bento_pb)
        svc = load_from_dir(saved_bundle_path)
        return svc

    def _upload_bento(self, bento_name, bento_version, saved_bento_bundle_path):
        try:
            streaming_request_generator = UploadBentoStreamRequests(
                bento_name=bento_name,
                bento_version=bento_version,
                bento_bundle_path=saved_bento_bundle_path,
            )
            result = self.yatai_service.UploadBento(
                iter(streaming_request_generator,),
                timeout=DEFAULT_GRPC_REQUEST_TIMEOUT,
            )
            if result.status.status_code != status_pb2.Status.OK:
                raise BentoMLException(result.status.error_message)
        except grpc.RpcError as e:
            raise BentoMLRpcError(
                e,
                f"Failed to upload {bento_name}:{bento_version} to "
                f"the remote yatai server",
            )
        finally:
            streaming_request_generator.close()

    def _download_bento(self, bento_name, bento_version):
        with TempDirectory(cleanup=False) as temp_dir:
            try:
                temp_tar_path = os.path.join(temp_dir, f"{uuid.uuid4().hex[:12]}.tar")
                response_iterator = self.yatai_service.DownloadBento(
                    DownloadBentoRequest(
                        bento_name=bento_name, bento_version=bento_version
                    ),
                    timeout=DEFAULT_GRPC_REQUEST_TIMEOUT,
                )
                with open(temp_tar_path, "wb+") as file:
                    for response in response_iterator:
                        if response.status.status_code != status_pb2.Status.OK:
                            raise BentoMLException(response.status.error_message)
                        file.write(response.bento_bundle)
                    file.seek(0)
                    temp_bundle_path = os.path.join(
                        temp_dir, f"{bento_name}_{bento_version}"
                    )
                    with tarfile.open(fileobj=file, mode="r") as tar:
                        tar.extractall(path=temp_bundle_path)
                return temp_bundle_path
            except grpc.RpcError as e:
                raise BentoMLRpcError(
                    e,
                    f"Failed to download {bento_name}:{bento_version} from "
                    f"the remote yatai server",
                )
