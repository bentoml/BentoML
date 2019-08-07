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
from bentoml.proto.status_pb2 import Status as StatusProto
from bentoml.deployment.operator import get_deployment_operator

from bentoml.deployment.store import DeploymentStore
from bentoml.exceptions import BentoMLException
from bentoml.proto.yatai_service_pb2_grpc import YataiServicer


LOG = logging.getLogger(__name__)

# pylint: disable=unused-argument
class YataiService(YataiServicer):
    def __init__(self):
        self.store = DeploymentStore()

    def HealthCheck(self, request, context):
        raise NotImplementedError('Method not implemented!')

    def GetYataiServiceVersion(self, request, context):
        raise NotImplementedError('Method not implemented!')

    def ApplyDeployment(self, request, context):
        try:
            deployment_pb = request.deployment
            operator = get_deployment_operator(deployment_pb)
            return operator.apply(request)

        except BentoMLException:
            response = ApplyDeploymentResponse()
            # response.status = ...
            # LOG.error(....)
            return response

    def DeleteDeployment(self, request, context):
        try:
            deployment_name = request.deployment_name
            deployment_pb = self.store.get(deployment_name)
            operator = get_deployment_operator(deployment_pb)
            return operator.delete(request)

        except BentoMLException:
            response = DeleteDeploymentResponse()
            # response.status = ...
            # LOG.error(....)
            return response

    def GetDeployment(self, request, context):
        # deployment_name = request.deployment_name
        # deployment_pb = self.store.get(deployment_name)
        # # get deployment status etc
        #
        # response = GetDeploymentResponse()
        # # construct deployment status into GetDeploymentResponse
        pass

    def DescribeDeployment(self, request, context):
        # deployment_name = request.deployment_name
        # response = DescribeDeploymentResponse()
        # # ...
        pass

    def ListDeployments(self, request, context):
        # deployment_pb_list = self.store.list(
        #     request.filter,
        #     request.labels,
        #     request.offset,
        #     request.limit,
        # )
        # response = ListDeploymentsResponse()
        pass

    def AddBento(self, request_iterator, context):
        raise NotImplementedError('Method not implemented!')

    def RemoveBento(self, request, context):
        raise NotImplementedError('Method not implemented!')

    def GetBento(self, request, context):
        raise NotImplementedError('Method not implemented!')

    def ListBento(self, request, context):
        raise NotImplementedError('Method not implemented!')


# pylint: enable=unused-argument

class Status(object):
    @staticmethod
    def OK(message=None):
        return StatusProto(status_code=StatusProto.OK, error_message=message)

    @staticmethod
    def CANCELLED(message=None):
        return StatusProto(status_code=StatusProto.CANCELLED, error_message=message)

    @staticmethod
    def UNKNOWN(message=None):
        return StatusProto(status_code=StatusProto.UNKNOWN, error_message=message)

    @staticmethod
    def INVALID_ARGUMENT(message=None):
        return StatusProto(
            status_code=StatusProto.INVALID_ARGUMENT, error_message=message
        )

    @staticmethod
    def DEADLINE_EXCEEDED(message=None):
        return StatusProto(
            status_code=StatusProto.DEADLINE_EXCEEDED, error_message=message
        )

    @staticmethod
    def NOT_FOUND(message=None):
        return StatusProto(status_code=StatusProto.NOT_FOUND, error_message=message)

    @staticmethod
    def ALREADY_EXISTS(message=None):
        return StatusProto(
            status_code=StatusProto.ALREADY_EXISTS, error_message=message
        )

    @staticmethod
    def PERMISSION_DENIED(message=None):
        return StatusProto(
            status_code=StatusProto.PERMISSION_DENIED, error_message=message
        )

    @staticmethod
    def UNAUTHENTICATED(message=None):
        return StatusProto(
            status_code=StatusProto.UNAUTHENTICATED, error_message=message
        )

    @staticmethod
    def RESOURCE_EXHAUSTED(message=None):
        return StatusProto(
            status_code=StatusProto.RESOURCE_EXHAUSTED, error_message=message
        )

    @staticmethod
    def FAILED_RECONDITION(message=None):
        return StatusProto(
            status_code=StatusProto.FAILED_PRECONDITION, error_message=message
        )

    @staticmethod
    def ABORTED(message=None):
        return StatusProto(status_code=StatusProto.ABORTED, error_message=message)

    @staticmethod
    def OUT_OF_RANGE(message=None):
        return StatusProto(status_code=StatusProto.OUT_OF_RANGE, error_message=message)

    @staticmethod
    def UNIMPLEMENTED(message=None):
        return StatusProto(status_code=StatusProto.UNIMPLEMENTED, error_message=message)

    @staticmethod
    def INTERNAL(message=None):
        return StatusProto(status_code=StatusProto.INTERNAL, error_message=message)

    @staticmethod
    def UNAVAILABLE(message=None):
        return StatusProto(status_code=StatusProto.UNAVAILABLE, error_message=message)

    @staticmethod
    def DATA_LOSS(message=None):
        return StatusProto(status_code=StatusProto.DATA_LOSS, error_message=message)