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
from bentoml.deployment.operator import get_deployment_operator
from bentoml.deployment.store import DeploymentStore
from bentoml.exceptions import BentoMLException
from bentoml.proto.yatai_service_pb2_grpc import YataiServicer
from bentoml.repository import get_default_repository
from bentoml.db import init_db
from bentoml.yatai.status import Status


logger = logging.getLogger(__name__)

# pylint: disable=unused-argument
class YataiService(YataiServicer):
    def __init__(self, db_config=None, bento_repository=None):
        self.sess_maker = init_db(db_config)
        self.deployment_store = DeploymentStore(self.sess_maker)
        self.repo = bento_repository or get_default_repository()

    def HealthCheck(self, request, context):
        raise NotImplementedError('Method not implemented!')

    def GetYataiServiceVersion(self, request, context):
        raise NotImplementedError('Method not implemented!')

    def ApplyDeployment(self, request, context):
        try:
            deployment_pb = request.deployment
            operator = get_deployment_operator(deployment_pb)
            return operator.apply(request)

        except BentoMLException as e:
            logger.error("INTERNAL ERROR:", e)
            return ApplyDeploymentResponse(Status.INTERNAL(e))

    def DeleteDeployment(self, request, context):
        try:
            deployment_name = request.deployment_name
            namespace = request.namespace
            deployment_pb = self.deployment_store.get(deployment_name, namespace)

            if deployment_pb:
                operator = get_deployment_operator(deployment_pb)
                return operator.delete(request)
            else:
                return DeleteDeploymentResponse(
                    status=Status.NOT_FOUND(
                        'Deployment "{}" in namespace "{}" not found'.format(
                            deployment_name, namespace
                        )
                    )
                )

        except BentoMLException as e:
            logger.error("INTERNAL ERROR:", e)
            return DeleteDeploymentResponse(status=Status.INTERNAL(e))

    def GetDeployment(self, request, context):
        try:
            deployment_name = request.deployment_name
            namespace = request.namespace
            deployment_pb = self.deployment_store.get(deployment_name, namespace)
            if deployment_pb:
                return GetDeploymentResponse(
                    status=Status.OK(), deployment=deployment_pb
                )
            else:
                return GetDeploymentResponse(
                    status=Status.NOT_FOUND(
                        'Deployment "{}" in namespace "{}" not found'.format(
                            deployment_name, namespace
                        )
                    )
                )
        except BentoMLException as e:
            logger.error("INTERNAL ERROR:", e)
            return GetDeploymentResponse(status=Status.INTERNAL(e))

    def DescribeDeployment(self, request, context):
        try:
            deployment_name = request.deployment_name
            namespace = request.namespace
            deployment_pb = self.deployment_store.get(deployment_name, namespace)

            if deployment_pb:
                operator = get_deployment_operator(deployment_pb)
                return operator.describe(deployment_pb)
            else:
                return DescribeDeploymentResponse(
                    status=Status.NOT_FOUND(
                        'Deployment "{}" in namespace "{}" not found'.format(
                            deployment_name, namespace
                        )
                    )
                )
        except BentoMLException as e:
            logger.error("INTERNAL ERROR:", e)
            return DescribeDeploymentResponse(Status.INTERNAL(e))

    def ListDeployments(self, request, context):
        try:
            deployment_pb_list = self.deployment_store.list(
                filter_str=request.filter,
                labels=MessageToDict(request.labels),
                offset=request.offset,
                limit=request.limit,
            )

            return ListDeploymentsResponse(
                status=Status.OK(), deployments=deployment_pb_list
            )
        except BentoMLException as e:
            logger.error("INTERNAL ERROR:", e)
            return ListDeploymentsResponse(status=Status.INTERNAL(e))

    def AddBento(self, request_iterator, context):
        raise NotImplementedError('Method not implemented!')

    def RemoveBento(self, request, context):
        raise NotImplementedError('Method not implemented!')

    def GetBento(self, request, context):
        raise NotImplementedError('Method not implemented!')

    def ListBento(self, request, context):
        raise NotImplementedError('Method not implemented!')


# pylint: enable=unused-argument
