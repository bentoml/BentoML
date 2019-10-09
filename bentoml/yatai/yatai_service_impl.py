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


import logging

from bentoml.proto.deployment_pb2 import (
    GetDeploymentResponse,
    DescribeDeploymentResponse,
    ListDeploymentsResponse,
    ApplyDeploymentResponse,
    DeleteDeploymentResponse,
    DeploymentState,
)
from bentoml.proto.repository_pb2 import (
    AddBentoResponse,
    DangerouslyDeleteBentoResponse,
    GetBentoResponse,
    UpdateBentoResponse,
    ListBentoResponse,
)
from bentoml.proto.yatai_service_pb2_grpc import YataiServicer
from bentoml.proto.yatai_service_pb2 import (
    HealthCheckResponse,
    GetYataiServiceVersionResponse,
)
from bentoml.deployment.operator import get_deployment_operator
from bentoml.deployment.store import DeploymentStore
from bentoml.exceptions import BentoMLException
from bentoml.repository import BentoRepository
from bentoml.repository.metadata_store import BentoMetadataStore
from bentoml.db import init_db
from bentoml.yatai.status import Status
from bentoml.proto import status_pb2
from bentoml.utils import ProtoMessageToDict
from bentoml.utils.validator import validate_deployment_pb_schema
from bentoml import __version__ as BENTOML_VERSION


logger = logging.getLogger(__name__)


class YataiService(YataiServicer):

    # pylint: disable=unused-argument

    def __init__(self, db_url, repo_base_url, default_namespace):
        self.default_namespace = default_namespace
        self.repo = BentoRepository(repo_base_url)
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

            validation_errors = validate_deployment_pb_schema(request.deployment)
            if validation_errors:
                return ApplyDeploymentResponse(
                    status=Status.ABORTED(
                        'Failed to validate deployment. {errors}'.format(
                            errors=validation_errors
                        )
                    )
                )

            previous_deployment = self.deployment_store.get(
                request.deployment.name, request.deployment.namespace
            )
            if previous_deployment:
                # check deployment platform
                if (
                    previous_deployment.spec.operator
                    != request.deployment.spec.operator
                ):
                    return ApplyDeploymentResponse(
                        status=Status.ABORTED(
                            'Can not change the target deploy platform of existing '
                            'active deployment. Try delete existing deployment and '
                            'deploy to new target platform again'
                        )
                    )
                request.deployment.state = DeploymentState(
                    state=DeploymentState.PENDING
                )

            self.deployment_store.insert_or_update(request.deployment)
            # find deployment operator based on deployment spec
            operator = get_deployment_operator(request.deployment)

            # deploying to target platform
            response = operator.apply(request.deployment, self, previous_deployment)

            # update deployment state
            self.deployment_store.insert_or_update(response.deployment)

            return response

        except BentoMLException as e:
            logger.error("INTERNAL ERROR: %s", e)
            return ApplyDeploymentResponse(status=Status.INTERNAL(str(e)))

    def DeleteDeployment(self, request, context=None):
        try:
            request.namespace = request.namespace or self.default_namespace
            deployment_pb = self.deployment_store.get(
                request.deployment_name, request.namespace
            )

            if deployment_pb:
                # find deployment operator based on deployment spec
                operator = get_deployment_operator(deployment_pb)

                # executing deployment deletion
                response = operator.delete(deployment_pb, self)

                # if delete successful, remove it from active deployment records table
                if response.status.status_code == status_pb2.Status.OK:
                    self.deployment_store.delete(
                        request.deployment_name, request.namespace
                    )
                    return response

                # If force delete flag is True, we will remove the record
                # from yatai database.
                if request.force_delete:
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
            logger.error("INTERNAL ERROR: %s", e)
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
            logger.error("INTERNAL ERROR: %s", e)
            return GetDeploymentResponse(status=Status.INTERNAL(str(e)))

    def DescribeDeployment(self, request, context=None):
        try:
            request.namespace = request.namespace or self.default_namespace
            deployment_pb = self.deployment_store.get(
                request.deployment_name, request.namespace
            )

            if deployment_pb:
                operator = get_deployment_operator(deployment_pb)

                response = operator.describe(deployment_pb, self)

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
            logger.error("INTERNAL ERROR: %s", e)
            return DescribeDeploymentResponse(Status.INTERNAL(str(e)))

    def ListDeployments(self, request, context=None):
        try:
            namespace = request.namespace or self.default_namespace
            deployment_pb_list = self.deployment_store.list(
                namespace=namespace,
                filter_str=request.filter,
                labels=request.labels,
                offset=request.offset,
                limit=request.limit,
            )

            return ListDeploymentsResponse(
                status=Status.OK(), deployments=deployment_pb_list
            )
        except BentoMLException as e:
            logger.error("INTERNAL ERROR: %s", e)
            return ListDeploymentsResponse(status=Status.INTERNAL(str(e)))

    def AddBento(self, request, context=None):
        try:
            # TODO: validate request
            new_bento_uri = self.repo.add(request.bento_name, request.bento_version)
            self.bento_metadata_store.add(
                bento_name=request.bento_name,
                bento_version=request.bento_version,
                uri=new_bento_uri.uri,
                uri_type=new_bento_uri.type,
            )
            return AddBentoResponse(status=Status.OK(), uri=new_bento_uri)
        except BentoMLException as e:
            logger.error("INTERNAL ERROR: %s", e)
            return AddBentoResponse(status=Status.INTERNAL(str(e)))

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
        except BentoMLException as e:
            logger.error("INTERNAL ERROR: %s", e)
            return UpdateBentoResponse(status=Status.INTERNAL(str(e)))

    def DangerouslyDeleteBento(self, request, context=None):
        try:
            # TODO: validate request
            self.bento_metadata_store.dangerously_delete(
                request.bento_name, request.bento_version
            )
        except BentoMLException as e:
            logger.error("INTERNAL ERROR: %s", e)
            return DangerouslyDeleteBentoResponse(status=Status.INTERNAL(str(e)))

    def GetBento(self, request, context=None):
        try:
            # TODO: validate request
            bento_metadata_pb = self.bento_metadata_store.get(
                request.bento_name, request.bento_version
            )
            if bento_metadata_pb:
                return GetBentoResponse(status=Status.OK(), bento=bento_metadata_pb)
            else:
                return GetBentoResponse(
                    status=Status.NOT_FOUND(
                        "Bento `{}:{}` is not found".format(
                            request.bento_name, request.bento_version
                        )
                    )
                )
        except BentoMLException as e:
            logger.error("INTERNAL ERROR: %s", e)
            return GetBentoResponse(status=Status.INTERNAL(str(e)))

    def ListBento(self, request, context=None):
        try:
            # TODO: validate request
            bento_metadata_pb_list = self.bento_metadata_store.list(
                request.bento_name, request.offset, request.limit, request.filter
            )

            return ListBentoResponse(status=Status.OK(), bentos=bento_metadata_pb_list)
        except BentoMLException as e:
            logger.error("INTERNAL ERROR: %s", e)
            return ListBentoResponse(status=Status.INTERNAL(str(e)))

    # pylint: enable=unused-argument
