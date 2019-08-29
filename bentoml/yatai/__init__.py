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

from google.protobuf.json_format import MessageToDict


from bentoml.proto.deployment_pb2 import (
    GetDeploymentResponse,
    DescribeDeploymentResponse,
    ListDeploymentsResponse,
    ApplyDeploymentResponse,
    DeleteDeploymentResponse,
)
from bentoml.proto.yatai_service_pb2_grpc import YataiServicer
from bentoml.proto.yatai_service_pb2 import (
    HealthCheckResponse,
    GetYataiServiceVersionResponse,
)
from bentoml.config import config
from bentoml.deployment.operator import get_deployment_operator
from bentoml.deployment.store import DeploymentStore
from bentoml.exceptions import BentoMLException
from bentoml.repository import get_default_repository
from bentoml.db import init_db
from bentoml.yatai.status import Status
from bentoml.proto import status_pb2
from bentoml import __version__ as BENTOML_VERSION


logger = logging.getLogger(__name__)


def get_yatai_service():
    return YataiService()


# pylint: disable=unused-argument
class YataiService(YataiServicer):
    def __init__(self, db_config=None, bento_repository=None, default_namespace=None):
        self.default_namespace = default_namespace or config.get(
            'deployment', 'default_namespace'
        )
        self.sess_maker = init_db(db_config)
        self.deployment_store = DeploymentStore(self.sess_maker)
        self.repo = bento_repository or get_default_repository()

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

            deployment_orm = self.deployment_store.get(
                request.deployment.name,
                request.deployment.namespace
            )
            if deployment_orm:
                # check deployment platform
                if deployment_orm.spec.operator != request.deployment.spec.operator:
                    return ApplyDeploymentResponse(status=Status.ABORTED('different platform'))
                previous_deployment = deployment_orm

                with self.deployment_store.update_deployment(
                        request.deployment.name, request.deployment.namespace
                ) as deployment:
                    deployment.spec = MessageToDict(request.deployment.spec)
            else:
                previous_deployment = None
                # create or update deployment spec record
                self.deployment_store.insert_or_update(request.deployment)

            # find deployment operator based on deployment spec
            operator = get_deployment_operator(request.deployment)

            # deploying to target platform
            response = operator.apply(request.deployment, self.repo, previous_deployment)

            # update deployment state
            self.deployment_store.insert_or_update(response.deployment)

            return response

        except BentoMLException as e:
            logger.error("INTERNAL ERROR: %s", e)
            return ApplyDeploymentResponse(status=Status.INTERNAL(e))

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
                response = operator.delete(deployment_pb, self.repo)

                # if delete successful, remove it from active deployment records table
                if response.status.status_code == status_pb2.Status.OK:
                    self.deployment_store.delete(
                        request.deployment_name, request.namespace
                    )

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
            return DeleteDeploymentResponse(status=Status.INTERNAL(e))

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
            return GetDeploymentResponse(status=Status.INTERNAL(e))

    def DescribeDeployment(self, request, context=None):
        try:
            request.namespace = request.namespace or self.default_namespace
            deployment_pb = self.deployment_store.get(
                request.deployment_name, request.namespace
            )

            if deployment_pb:
                operator = get_deployment_operator(deployment_pb)
                response = operator.describe(deployment_pb, self.repo)

                if response.status.status_code == status_pb2.Status.OK:
                    with self.deployment_store.update_deployment(
                        request.deployment_name, request.namespace
                    ) as deployment:
                        deployment.state = MessageToDict(response.state)

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
            return DescribeDeploymentResponse(Status.INTERNAL(e))

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
            return ListDeploymentsResponse(status=Status.INTERNAL(e))

    def AddBento(self, request_iterator, context=None):
        raise NotImplementedError('Method not implemented!')

    def RemoveBento(self, request, context=None):
        raise NotImplementedError('Method not implemented!')

    def GetBento(self, request, context=None):
        raise NotImplementedError('Method not implemented!')

    def ListBento(self, request, context=None):
        raise NotImplementedError('Method not implemented!')


# pylint: enable=unused-argument
