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
from datetime import datetime
import logging

from bentoml import config
from bentoml.utils.usage_stats import track
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
)
from bentoml.yatai.proto.yatai_service_pb2_grpc import YataiServicer
from bentoml.yatai.proto.yatai_service_pb2 import (
    HealthCheckResponse,
    GetYataiServiceVersionResponse,
)
from bentoml.yatai.deployment.operator import get_deployment_operator
from bentoml.yatai.deployment.store import DeploymentStore
from bentoml.exceptions import BentoMLException, InvalidArgument
from bentoml.yatai.repository.repository import Repository
from bentoml.yatai.repository.metadata_store import BentoMetadataStore
from bentoml.yatai.db import init_db
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


class YataiService(YataiServicer):

    # pylint: disable=unused-argument
    # pylint: disable=broad-except

    def __init__(
        self,
        db_url=None,
        repo_base_url=None,
        s3_endpoint_url=None,
        default_namespace=None,
    ):
        cfg = config('yatai_service')
        repo_base_url = repo_base_url or cfg.get('repository_base_url')
        db_url = db_url or cfg.get('db_url')
        s3_endpoint_url = s3_endpoint_url or cfg.get('s3_endpoint_url') or None
        default_namespace = default_namespace or cfg.get('default_namespace')

        self.default_namespace = default_namespace
        self.repo = Repository(repo_base_url, s3_endpoint_url)
        self.sess_maker = init_db(db_url)
        self.deployment_store = DeploymentStore(self.sess_maker)
        self.bento_metadata_store = BentoMetadataStore(self.sess_maker)

    def HealthCheck(self, request, context=None):
        return HealthCheckResponse(status=Status.OK())

    def GetYataiServiceVersion(self, request, context=None):
        return GetYataiServiceVersionResponse(status=Status.OK, version=BENTOML_VERSION)

    def ApplyDeployment(self, request, context=None):
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

            previous_deployment = self.deployment_store.get(
                request.deployment.name, request.deployment.namespace
            )
            if not previous_deployment:
                request.deployment.created_at.GetCurrentTime()
            request.deployment.last_updated_at.GetCurrentTime()

            self.deployment_store.insert_or_update(request.deployment)
            # find deployment operator based on deployment spec
            operator = get_deployment_operator(self, request.deployment)

            # deploying to target platform
            if previous_deployment:
                response = operator.update(request.deployment, previous_deployment)
            else:
                response = operator.add(request.deployment)

            if response.status.status_code == status_pb2.Status.OK:
                # update deployment state
                if response and response.deployment:
                    self.deployment_store.insert_or_update(response.deployment)
                else:
                    raise BentoMLException(
                        "DeploymentOperator Internal Error: failed to add or update "
                        "deployment metadata to database"
                    )
                logger.info(
                    "ApplyDeployment (%s, namespace %s) succeeded",
                    request.deployment.name,
                    request.deployment.namespace,
                )
            else:
                if not previous_deployment:
                    # When failed to create the deployment, delete it from active
                    # deployments records
                    self.deployment_store.delete(
                        request.deployment.name, request.deployment.namespace
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
        try:
            request.namespace = request.namespace or self.default_namespace
            deployment_pb = self.deployment_store.get(
                request.deployment_name, request.namespace
            )

            if deployment_pb:
                # find deployment operator based on deployment spec
                operator = get_deployment_operator(self, deployment_pb)

                # executing deployment deletion
                response = operator.delete(deployment_pb)

                # if delete successful, remove it from active deployment records table
                if response.status.status_code == status_pb2.Status.OK:
                    track_deployment_delete(
                        deployment_pb.spec.operator, deployment_pb.created_at
                    )
                    self.deployment_store.delete(
                        request.deployment_name, request.namespace
                    )
                    return response

                # If force delete flag is True, we will remove the record
                # from yatai database.
                if request.force_delete:
                    # Track deployment delete before it vanquishes from deployment store
                    track_deployment_delete(
                        deployment_pb.spec.operator, deployment_pb.created_at, True
                    )
                    self.deployment_store.delete(
                        request.deployment_name, request.namespace
                    )
                    return DeleteDeploymentResponse(status=Status.OK())

                if response.status.status_code == status_pb2.Status.NOT_FOUND:
                    modified_message = (
                        'Cloud resources not found, error: {} - it may have been '
                        'deleted manually. Try delete deployment '
                        'with "--force" option to ignore this error '
                        'and force deleting the deployment record'.format(
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
        try:
            request.namespace = request.namespace or self.default_namespace
            deployment_pb = self.deployment_store.get(
                request.deployment_name, request.namespace
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
        try:
            request.namespace = request.namespace or self.default_namespace
            deployment_pb = self.deployment_store.get(
                request.deployment_name, request.namespace
            )

            if deployment_pb:
                operator = get_deployment_operator(self, deployment_pb)

                response = operator.describe(deployment_pb)

                if response.status.status_code == status_pb2.Status.OK:
                    with self.deployment_store.update_deployment(
                        request.deployment_name, request.namespace
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
        try:
            namespace = request.namespace or self.default_namespace
            deployment_pb_list = self.deployment_store.list(
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
        try:
            # TODO: validate request
            bento_pb = self.bento_metadata_store.get(
                request.bento_name, request.bento_version
            )
            if bento_pb:
                error_message = "BentoService bundle: {}:{} already exist".format(
                    request.bento_name, request.bento_version
                )
                logger.error(error_message)
                return AddBentoResponse(status=Status.ABORTED(error_message))
            new_bento_uri = self.repo.add(request.bento_name, request.bento_version)
            self.bento_metadata_store.add(
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
        try:
            # TODO: validate request
            if request.upload_status:
                self.bento_metadata_store.update_upload_status(
                    request.bento_name, request.bento_version, request.upload_status
                )
            if request.service_metadata:
                self.bento_metadata_store.update_bento_service_metadata(
                    request.bento_name, request.bento_version, request.service_metadata
                )
            return UpdateBentoResponse(status=Status.OK())
        except BentoMLException as e:
            logger.error("RPC ERROR UpdateBento: %s", e)
            return UpdateBentoResponse(status=e.status_proto)
        except Exception as e:  # pylint: disable=broad-except
            logger.error("RPC ERROR UpdateBento: %s", e)
            return UpdateBentoResponse(status=Status.INTERNAL())

    def DangerouslyDeleteBento(self, request, context=None):
        try:
            # TODO: validate request
            bento_pb = self.bento_metadata_store.get(
                request.bento_name, request.bento_version
            )
            if not bento_pb:
                msg = (
                    f"BentoService {request.bento_name}:{request.bento_version} "
                    f"has already been deleted"
                )
                return DangerouslyDeleteBentoResponse(status=Status.ABORTED(msg))

            logger.debug(
                'Deleting BentoService %s:%s', request.bento_name, request.bento_version
            )
            self.bento_metadata_store.dangerously_delete(
                request.bento_name, request.bento_version
            )
            self.repo.dangerously_delete(request.bento_name, request.bento_version)
            return DangerouslyDeleteBentoResponse(status=Status.OK())
        except BentoMLException as e:
            logger.error("RPC ERROR DangerouslyDeleteBento: %s", e)
            return DangerouslyDeleteBentoResponse(status=e.status_proto)
        except Exception as e:  # pylint: disable=broad-except
            logger.error("RPC ERROR DangerouslyDeleteBento: %s", e)
            return DangerouslyDeleteBentoResponse(status=Status.INTERNAL())

    def GetBento(self, request, context=None):
        try:
            # TODO: validate request
            bento_pb = self.bento_metadata_store.get(
                request.bento_name, request.bento_version
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
        try:
            # TODO: validate request
            bento_metadata_pb_list = self.bento_metadata_store.list(
                bento_name=request.bento_name,
                offset=request.offset,
                limit=request.limit,
                order_by=request.order_by,
                label_selectors=request.label_selectors,
                ascending_order=request.ascending_order,
            )

            return ListBentoResponse(status=Status.OK(), bentos=bento_metadata_pb_list)
        except BentoMLException as e:
            logger.error("RPC ERROR ListBento: %s", e)
            return ListBentoResponse(status=e.status_proto)
        except Exception as e:  # pylint: disable=broad-except
            logger.error("RPC ERROR ListBento: %s", e)
            return ListBentoResponse(status=Status.INTERNAL())

    # pylint: enable=unused-argument
