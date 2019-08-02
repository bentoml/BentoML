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
    # GetDeploymentResponse,
    # DescribeDeploymentResponse,
    # ListDeploymentsResponse,
    ApplyDeploymentResponse,
    DeleteDeploymentResponse,
)
from bentoml.deployment.operator import get_deployment_operator

from bentoml.deployment.store import DeploymentStore
from bentoml.exceptions import BentoMLException


LOG = logging.getLogger(__name__)


class DeploymentService(object):
    def __init__(self):
        self.store = DeploymentStore()

    def apply(self, apply_deployment_request):
        try:
            deployment_pb = apply_deployment_request.deployment
            operator = get_deployment_operator(deployment_pb)
            return operator.apply(apply_deployment_request)

        except BentoMLException:
            response = ApplyDeploymentResponse()
            # response.status = ...
            # LOG.error(....)
            return response

    def delete(self, delete_deployment_request):
        try:
            deployment_name = delete_deployment_request.deployment_name
            deployment_pb = self.store.get(deployment_name)
            operator = get_deployment_operator(deployment_pb)
            return operator.delete(delete_deployment_request)

        except BentoMLException:
            response = DeleteDeploymentResponse()
            # response.status = ...
            # LOG.error(....)
            return response

    def get(self, get_deployment_request):
        # deployment_name = get_deployment_request.deployment_name
        # deployment_pb = self.store.get(deployment_name)
        # # get deployment status etc
        #
        # response = GetDeploymentResponse()
        # # construct deployment status into GetDeploymentResponse
        pass

    def describe(self, describe_deployment_request):
        # deployment_name = describe_deployment_request.deployment_name
        # response = DescribeDeploymentResponse()
        # # ...
        pass

    def list(self, list_deployments_request):
        # deployment_pb_list = self.store.list(
        #     list_deployments_request.filter,
        #     list_deployments_request.labels,
        #     list_deployments_request.offset,
        #     list_deployments_request.limit,
        # )
        # response = ListDeploymentsResponse()
        pass
