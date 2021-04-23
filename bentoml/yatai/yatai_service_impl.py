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
import math
import os
import shutil
import tarfile
import uuid
from datetime import datetime
import logging
from datetime import datetime
from dependency_injector.wiring import Provide, inject

from bentoml.configuration.containers import BentoMLContainer
from bentoml.saved_bundle import safe_retrieve
from bentoml.utils.docker_utils import (
    to_valid_docker_image_name,
    to_valid_docker_image_version,
)
from bentoml.utils.tempdir import TempDirectory
from bentoml.utils.usage_stats import track
from bentoml.yatai.deployment.docker_utils import ensure_docker_available_or_raise
from bentoml.yatai.locking.lock import LockType, lock
from bentoml.yatai.proto.deployment_pb2 import (
    GetDeploymentResponse,
    DescribeDeploymentResponse,
    ListDeploymentsResponse,
    ApplyDeploymentResponse,
    DeleteDeploymentResponse,
    DeploymentSpec,
)
from bentoml.yatai.proto.repository_pb2 import (
    AddBentoResponse,
    DangerouslyDeleteBentoResponse,
    GetBentoResponse,
    UpdateBentoResponse,
    ListBentoResponse,
    BentoUri,
    ContainerizeBentoResponse,
    UploadBentoResponse,
    DownloadBentoResponse,
)
from bentoml.yatai.proto.yatai_service_pb2 import (
    HealthCheckResponse,
    GetYataiServiceVersionResponse,
)
from bentoml.yatai.deployment.operator import get_deployment_operator
from bentoml.exceptions import (
    BentoMLException,
    InvalidArgument,
    YataiRepositoryException,
)
from bentoml.yatai.repository.base_repository import BaseRepository
from bentoml.yatai.db import DB
from bentoml.yatai.status import Status
from bentoml.yatai.proto import status_pb2
from bentoml.utils import ProtoMessageToDict
from bentoml.yatai.validator import validate_deployment_pb
from bentoml import __version__ as BENTOML_VERSION

logger = logging.getLogger(__name__)


def track_deployment_delete(deployment_operator, created_at, force_delete=False):
    operator_name = DeploymentSpec.DeploymentOperator.Name(deployment_operator)
    up_time = int((datetime.utcnow() - created_at.ToDatetime()).total_seconds())
    track(
        f'deployment-{operator_name}-stop',
        {'up_time': up_time, 'force_delete': force_delete},
    )


def get_yatai_service_impl(base=object):
    # This helps avoid loading YataiServicer grpc file when using local YataiService
    # implementation. This speeds up regular save/load operations when Yatai is not used

    class YataiServiceImpl(base):

        # pylint: disable=unused-argument
        # pylint: disable=broad-except
        @inject
        def __init__(
            self,
            database: DB,
            repository: BaseRepository,
            default_namespace: str = Provide[BentoMLContainer.config.yatai.namespace],
        ):
            self.default_namespace = default_namespace
            self.repo = repository
            self.db = database

        def HealthCheck(self, request, context=None):
            return HealthCheckResponse(status=Status.OK())

        def GetYataiServiceVersion(self, request, context=None):
            return GetYataiServiceVersionResponse(
                status=Status.OK, version=BENTOML_VERSION
            )

        def ApplyDeployment(self, request, context=None):
            deployment_id = f"{request.deployment.name}_{request.deployment.namespace}"
            spec = request.deployment.spec
            bento_id = f"{spec.bento_name}_{spec.bento_version}"
            with lock(
                self.db, [(deployment_id, LockType.WRITE), (bento_id, LockType.READ)]
            ) as (
                sess,
                _,
            ):
                # to bento level lock
                try:
                    # apply default namespace if not set
                    request.deployment.namespace = (
                        request.deployment.namespace or self.default_namespace
                    )

                    validation_errors = validate_deployment_pb(request.deployment)
                    if validation_errors:
                        raise InvalidArgument(
                            'Failed to validate deployment. {errors}'.format(
                                errors=validation_errors
                            )
                        )

                    previous_deployment = self.db.deployment_store.get(
                        sess, request.deployment.name, request.deployment.namespace
                    )
                    if not previous_deployment:
                        request.deployment.created_at.GetCurrentTime()
                    request.deployment.last_updated_at.GetCurrentTime()

                    self.db.deployment_store.insert_or_update(sess, request.deployment)
                    # find deployment operator based on deployment spec
                    operator = get_deployment_operator(self, request.deployment)

                    # deploying to target platform
                    if previous_deployment:
                        response = operator.update(
                            request.deployment, previous_deployment
                        )
                    else:
                        response = operator.add(request.deployment)

                    if response.status.status_code == status_pb2.Status.OK:
                        # update deployment state
                        if response and response.deployment:
                            self.db.deployment_store.insert_or_update(
                                sess, response.deployment
                            )
                        else:
                            raise BentoMLException(
                                "DeploymentOperator Internal Error: failed to add or "
                                "update deployment metadata to database"
                            )
                        logger.info(
                            "ApplyDeployment (%s, namespace %s) succeeded",
                            request.deployment.name,
                            request.deployment.namespace,
                        )
                    else:
                        if not previous_deployment:
                            # When failed to create the deployment,
                            # delete it from active deployments records
                            self.db.deployment_store.delete(
                                sess,
                                request.deployment.name,
                                request.deployment.namespace,
                            )
                        logger.debug(
                            "ApplyDeployment (%s, namespace %s) failed: %s",
                            request.deployment.name,
                            request.deployment.namespace,
                            response.status.error_message,
                        )

                    return response

                except BentoMLException as e:
                    logger.error("RPC ERROR ApplyDeployment: %s", e)
                    return ApplyDeploymentResponse(status=e.status_proto)
                except Exception as e:
                    logger.error("URPC ERROR ApplyDeployment: %s", e)
                    return ApplyDeploymentResponse(status=Status.INTERNAL(str(e)))

        def DeleteDeployment(self, request, context=None):
            deployment_id = f"{request.deployment_name}_{request.namespace}"
            with lock(self.db, [(deployment_id, LockType.WRITE)]) as (
                sess,
                _,
            ):
                try:
                    request.namespace = request.namespace or self.default_namespace
                    deployment_pb = self.db.deployment_store.get(
                        sess, request.deployment_name, request.namespace
                    )

                    if deployment_pb:
                        # find deployment operator based on deployment spec
                        operator = get_deployment_operator(self, deployment_pb)

                        # executing deployment deletion
                        response = operator.delete(deployment_pb)

                        # if delete successful, remove it from active deployment records
                        # table
                        if response.status.status_code == status_pb2.Status.OK:
                            track_deployment_delete(
                                deployment_pb.spec.operator, deployment_pb.created_at
                            )
                            self.db.deployment_store.delete(
                                sess, request.deployment_name, request.namespace
                            )
                            return response

                        # If force delete flag is True, we will remove the record
                        # from yatai database.
                        if request.force_delete:
                            # Track deployment delete before it
                            # vanishes from deployment store
                            track_deployment_delete(
                                deployment_pb.spec.operator,
                                deployment_pb.created_at,
                                True,
                            )
                            self.db.deployment_store.delete(
                                sess, request.deployment_name, request.namespace
                            )
                            return DeleteDeploymentResponse(status=Status.OK())

                        if response.status.status_code == status_pb2.Status.NOT_FOUND:
                            modified_message = (
                                'Cloud resources not found, error: {} - it '
                                'may have been deleted manually. Try delete '
                                'deployment with "--force" option to ignore this '
                                'error and force deleting the deployment record'.format(
                                    response.status.error_message
                                )
                            )
                            response.status.error_message = modified_message

                        return response
                    else:
                        return DeleteDeploymentResponse(
                            status=Status.NOT_FOUND(
                                'Deployment "{}" in namespace "{}" not found'.format(
                                    request.deployment_name, request.namespace
                                )
                            )
                        )

                except BentoMLException as e:
                    logger.error("RPC ERROR DeleteDeployment: %s", e)
                    return DeleteDeploymentResponse(status=e.status_proto)
                except Exception as e:  # pylint: disable=broad-except
                    logger.error("RPC ERROR DeleteDeployment: %s", e)
                    return DeleteDeploymentResponse(status=Status.INTERNAL(str(e)))

        def GetDeployment(self, request, context=None):
            deployment_id = f"{request.deployment_name}_{request.namespace}"
            with lock(self.db, [(deployment_id, LockType.READ)]) as (
                sess,
                _,
            ):
                try:
                    request.namespace = request.namespace or self.default_namespace
                    deployment_pb = self.db.deployment_store.get(
                        sess, request.deployment_name, request.namespace
                    )
                    if deployment_pb:
                        return GetDeploymentResponse(
                            status=Status.OK(), deployment=deployment_pb
                        )
                    else:
                        return GetDeploymentResponse(
                            status=Status.NOT_FOUND(
                                'Deployment "{}" in namespace "{}" not found'.format(
                                    request.deployment_name, request.namespace
                                )
                            )
                        )
                except BentoMLException as e:
                    logger.error("RPC ERROR GetDeployment: %s", e)
                    return GetDeploymentResponse(status=e.status_proto)
                except Exception as e:  # pylint: disable=broad-except
                    logger.error("RPC ERROR GetDeployment: %s", e)
                    return GetDeploymentResponse(status=Status.INTERNAL())

        def DescribeDeployment(self, request, context=None):
            deployment_id = f"{request.deployment_name}_{request.namespace}"
            with lock(self.db, [(deployment_id, LockType.READ)]) as (
                sess,
                _,
            ):
                try:
                    request.namespace = request.namespace or self.default_namespace
                    deployment_pb = self.db.deployment_store.get(
                        sess, request.deployment_name, request.namespace
                    )

                    if deployment_pb:
                        operator = get_deployment_operator(self, deployment_pb)

                        response = operator.describe(deployment_pb)

                        if response.status.status_code == status_pb2.Status.OK:
                            with self.db.deployment_store.update_deployment(
                                sess, request.deployment_name, request.namespace
                            ) as deployment:
                                deployment.state = ProtoMessageToDict(response.state)

                        return response
                    else:
                        return DescribeDeploymentResponse(
                            status=Status.NOT_FOUND(
                                'Deployment "{}" in namespace "{}" not found'.format(
                                    request.deployment_name, request.namespace
                                )
                            )
                        )
                except BentoMLException as e:
                    logger.error("RPC ERROR DescribeDeployment: %s", e)
                    return DeleteDeploymentResponse(status=e.status_proto)
                except Exception as e:  # pylint: disable=broad-except
                    logger.error("RPC ERROR DescribeDeployment: %s", e)
                    return DeleteDeploymentResponse(status=Status.INTERNAL())

        def ListDeployments(self, request, context=None):
            with self.db.create_session() as sess:
                try:
                    namespace = request.namespace or self.default_namespace
                    deployment_pb_list = self.db.deployment_store.list(
                        sess,
                        namespace=namespace,
                        label_selectors=request.label_selectors,
                        offset=request.offset,
                        limit=request.limit,
                        operator=request.operator,
                        order_by=request.order_by,
                        ascending_order=request.ascending_order,
                    )

                    return ListDeploymentsResponse(
                        status=Status.OK(), deployments=deployment_pb_list
                    )
                except BentoMLException as e:
                    logger.error("RPC ERROR ListDeployments: %s", e)
                    return ListDeploymentsResponse(status=e.status_proto)
                except Exception as e:  # pylint: disable=broad-except
                    logger.error("RPC ERROR ListDeployments: %s", e)
                    return ListDeploymentsResponse(status=Status.INTERNAL())

        def AddBento(self, request, context=None):
            bento_id = f"{request.bento_name}_{request.bento_version}"
            with lock(self.db, [(bento_id, LockType.WRITE)]) as (sess, _):
                try:
                    # TODO: validate request
                    bento_pb = self.db.metadata_store.get(
                        sess, request.bento_name, request.bento_version
                    )
                    if bento_pb:
                        error_message = (
                            "BentoService bundle: "
                            "{}:{} already exists".format(
                                request.bento_name, request.bento_version
                            )
                        )
                        logger.error(error_message)
                        return AddBentoResponse(status=Status.ABORTED(error_message))
                    new_bento_uri = self.repo.add(
                        request.bento_name, request.bento_version
                    )
                    self.db.metadata_store.add(
                        sess,
                        bento_name=request.bento_name,
                        bento_version=request.bento_version,
                        uri=new_bento_uri.uri,
                        uri_type=new_bento_uri.type,
                    )
                    return AddBentoResponse(status=Status.OK(), uri=new_bento_uri)
                except BentoMLException as e:
                    logger.error("RPC ERROR AddBento: %s", e)
                    return DeleteDeploymentResponse(status=e.status_proto)
                except Exception as e:  # pylint: disable=broad-except
                    logger.error("URPC ERROR AddBento: %s", e)
                    return DeleteDeploymentResponse(status=Status.INTERNAL())

        def UpdateBento(self, request, context=None):
            bento_id = f"{request.bento_name}_{request.bento_version}"
            with lock(self.db, [(bento_id, LockType.WRITE)]) as (sess, _):
                try:
                    # TODO: validate request
                    if request.upload_status:
                        self.db.metadata_store.update_upload_status(
                            sess,
                            request.bento_name,
                            request.bento_version,
                            request.upload_status,
                        )
                    if request.service_metadata:
                        self.db.metadata_store.update(
                            sess,
                            request.bento_name,
                            request.bento_version,
                            request.service_metadata,
                        )
                    return UpdateBentoResponse(status=Status.OK())
                except BentoMLException as e:
                    logger.error("RPC ERROR UpdateBento: %s", e)
                    return UpdateBentoResponse(status=e.status_proto)
                except Exception as e:  # pylint: disable=broad-except
                    logger.error("RPC ERROR UpdateBento: %s", e)
                    return UpdateBentoResponse(status=Status.INTERNAL())

        def DangerouslyDeleteBento(self, request, context=None):
            bento_id = f"{request.bento_name}_{request.bento_version}"
            with lock(self.db, [(bento_id, LockType.WRITE)]) as (sess, _):
                try:
                    # TODO: validate request
                    bento_pb = self.db.metadata_store.get(
                        sess, request.bento_name, request.bento_version
                    )
                    if not bento_pb:
                        msg = (
                            f"BentoService "
                            f"{request.bento_name}:{request.bento_version} "
                            f"has already been deleted"
                        )
                        return DangerouslyDeleteBentoResponse(
                            status=Status.ABORTED(msg)
                        )

                    logger.debug(
                        'Deleting BentoService %s:%s',
                        request.bento_name,
                        request.bento_version,
                    )
                    self.db.metadata_store.dangerously_delete(
                        sess, request.bento_name, request.bento_version
                    )
                    self.repo.dangerously_delete(
                        request.bento_name, request.bento_version
                    )
                    return DangerouslyDeleteBentoResponse(status=Status.OK())
                except BentoMLException as e:
                    logger.error("RPC ERROR DangerouslyDeleteBento: %s", e)
                    return DangerouslyDeleteBentoResponse(status=e.status_proto)
                except Exception as e:  # pylint: disable=broad-except
                    logger.error("RPC ERROR DangerouslyDeleteBento: %s", e)
                    return DangerouslyDeleteBentoResponse(status=Status.INTERNAL())

        def GetBento(self, request, context=None):
            bento_id = f"{request.bento_name}_{request.bento_version}"
            with lock(self.db, [(bento_id, LockType.READ)]) as (sess, _):
                try:
                    # TODO: validate request
                    bento_pb = self.db.metadata_store.get(
                        sess, request.bento_name, request.bento_version
                    )
                    if bento_pb:
                        if request.bento_version.lower() == 'latest':
                            logger.info(
                                'Getting latest version %s:%s',
                                request.bento_name,
                                bento_pb.version,
                            )
                        if bento_pb.uri.type == BentoUri.S3:
                            bento_pb.uri.s3_presigned_url = self.repo.get(
                                bento_pb.name, bento_pb.version
                            )
                        elif bento_pb.uri.type == BentoUri.GCS:
                            bento_pb.uri.gcs_presigned_url = self.repo.get(
                                bento_pb.name, bento_pb.version
                            )
                        return GetBentoResponse(status=Status.OK(), bento=bento_pb)
                    else:
                        return GetBentoResponse(
                            status=Status.NOT_FOUND(
                                "BentoService `{}:{}` is not found".format(
                                    request.bento_name, request.bento_version
                                )
                            )
                        )
                except BentoMLException as e:
                    logger.error("RPC ERROR GetBento: %s", e)
                    return GetBentoResponse(status=e.status_proto)
                except Exception as e:  # pylint: disable=broad-except
                    logger.error("RPC ERROR GetBento: %s", e)
                    return GetBentoResponse(status=Status.INTERNAL())

        def ListBento(self, request, context=None):
            with self.db.create_session() as sess:
                try:
                    # TODO: validate request
                    bento_metadata_pb_list = self.db.metadata_store.list(
                        sess,
                        bento_name=request.bento_name,
                        offset=request.offset,
                        limit=request.limit,
                        order_by=request.order_by,
                        label_selectors=request.label_selectors,
                        ascending_order=request.ascending_order,
                    )

                    return ListBentoResponse(
                        status=Status.OK(), bentos=bento_metadata_pb_list
                    )
                except BentoMLException as e:
                    logger.error("RPC ERROR ListBento: %s", e)
                    return ListBentoResponse(status=e.status_proto)
                except Exception as e:  # pylint: disable=broad-except
                    logger.error("RPC ERROR ListBento: %s", e)
                    return ListBentoResponse(status=Status.INTERNAL())

        def ContainerizeBento(self, request, context=None):
            bento_id = f"{request.bento_name}_{request.bento_version}"
            with lock(self.db, [(bento_id, LockType.READ)]) as (sess, _):
                try:
                    ensure_docker_available_or_raise()
                    tag = request.tag
                    if tag is None or len(tag) == 0:
                        name = to_valid_docker_image_name(request.bento_name)
                        version = to_valid_docker_image_version(request.bento_version)
                        tag = f"{name}:{version}"
                    if ":" not in tag:
                        version = to_valid_docker_image_version(request.bento_version)
                        tag = f"{tag}:{version}"
                    import docker

                    docker_client = docker.from_env()
                    bento_pb = self.db.metadata_store.get(
                        sess, request.bento_name, request.bento_version
                    )
                    if not bento_pb:
                        raise YataiRepositoryException(
                            f'BentoService '
                            f'{request.bento_name}:{request.bento_version} '
                            f'does not exist'
                        )

                    with TempDirectory() as temp_dir:
                        temp_bundle_path = f'{temp_dir}/{bento_pb.name}'
                        bento_service_bundle_path = bento_pb.uri.uri
                        if bento_pb.uri.type == BentoUri.S3:
                            bento_service_bundle_path = self.repo.get(
                                bento_pb.name, bento_pb.version
                            )
                        elif bento_pb.uri.type == BentoUri.GCS:
                            bento_service_bundle_path = self.repo.get(
                                bento_pb.name, bento_pb.version
                            )
                        safe_retrieve(bento_service_bundle_path, temp_bundle_path)
                        try:
                            docker_client.images.build(
                                path=temp_bundle_path,
                                tag=tag,
                                buildargs=dict(request.build_args),
                            )
                        except (
                            docker.errors.APIError,
                            docker.errors.BuildError,
                        ) as error:
                            logger.error(f'Encounter container building issue: {error}')
                            raise YataiRepositoryException(error)
                        if request.push is True:
                            try:
                                docker_client.images.push(
                                    repository=request.repository, tag=tag
                                )
                            except docker.errors.APIError as error:
                                raise YataiRepositoryException(error)
                        return ContainerizeBentoResponse(status=Status.OK(), tag=tag)
                except BentoMLException as e:
                    logger.error(f"RPC ERROR ContainerizeBento: {e}")
                    return ContainerizeBentoResponse(status=e.status_proto)
                except Exception as e:  # pylint: disable=broad-except
                    logger.error(f"RPC ERROR ContainerizeBento: {e}")
                    return ContainerizeBentoResponse(status=Status.INTERNAL(e))

        def UploadBento(self, request_iterator, context=None):
            # 1. create temp directory for the file
            # 2. write bytes into the tar file
            # 3. unzip tar to the fs storage
            # 4. update the path for the bento
            # 5. return response
            with self.db.create_session() as sess:
                try:
                    with TempDirectory() as temp_dir:
                        temp_tar_path = os.path.join(
                            temp_dir, f'{uuid.uuid4().hex[:12]}.tar'
                        )
                        print(f'saved yatai tar path: {temp_tar_path}')
                        file = open(temp_tar_path, 'wb+')
                        for request in request_iterator:
                            bento_name = request.bento_name
                            bento_version = request.bento_version
                            file.write(request.bento_bundle)
                        if bento_name is None or bento_version is None:
                            return UploadBentoResponse(status=Status.ABORTED())
                        bento_pb = self.db.metadata_store.get(
                            sess, bento_name, bento_version
                        )
                        file.seek(0)
                        if bento_pb:
                            # save the content
                            if os.path.exists(bento_pb.uri.uri):
                                shutil.rmtree(bento_pb.uri.uri)
                            tar = tarfile.open(fileobj=file, mode='r:gz')
                            # the tar is build an extra directory...
                            bento_bundle_root, _ = os.path.split(bento_pb.uri.uri)
                            tar.extractall(path=bento_bundle_root)
                            tar.close()
                            file.close()
                            return UploadBentoResponse(status=Status.OK())
                        else:
                            return UploadBentoResponse(
                                status=Status.NOT_FOUND(
                                    "BentoService `{}:{}` is not found".format(
                                        bento_name, bento_version
                                    )
                                )
                            )
                except BentoMLException as e:
                    logger.error("RPC ERROR GetBento: %s", e)
                    return UploadBentoResponse(status=e.status_proto)
                except Exception as e:  # pylint: disable=broad-except
                    logger.error("RPC ERROR GetBento: %s", e)
                    return UploadBentoResponse(status=Status.INTERNAL())

        def DownloadBento(self, request, context=None):
            with self.db.create_session() as sess:
                try:
                    bento_pb = self.db.metadata_store.get(
                        sess, request.bento_name, request.bento_version
                    )
                    with TempDirectory() as temp_dir:
                        tarfile_path = os.path.join(
                            temp_dir, f'{bento_pb.name}_{bento_pb.version}.tar'
                        )
                        file = open(tarfile_path, 'wb+')
                        with tarfile.open(mode='w:gz', fileobj=file) as tar:
                            tar.add(
                                bento_pb.uri.uri,
                                arcname=f'{bento_pb.name}_{bento_pb.version}',
                            )
                        file.seek(0)
                        file_size = os.path.getsize(tarfile_path)
                        chunk_size = 1024 * 1024  # 1M
                        chunk_count = math.ceil(float(file_size) / chunk_size)
                        sent_chunk_count = 0
                        file_index = 0
                        while sent_chunk_count < chunk_count:
                            current_file_end = min(file_size, file_index + chunk_size)
                            file.seek(file_index)
                            chunk = file.read(chunk_size)
                            file_index = current_file_end
                            sent_chunk_count += 1
                            response = DownloadBentoResponse(bento_bundle=chunk)
                            yield response
                        file.close()
                except BentoMLException as e:
                    logger.error("RPC ERROR GetBento: %s", e)
                    return DownloadBentoResponse(status=e.status_proto)
                except Exception as e:  # pylint: disable=broad-except
                    logger.error("RPC ERROR GetBento: %s", e)
                    return DownloadBentoResponse(status=Status.INTERNAL())

    # pylint: enable=unused-argument

    return YataiServiceImpl
