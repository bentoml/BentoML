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

from abc import abstractmethod, ABCMeta

from bentoml.proto.deployment_pb2 import DeploymentSpec
from bentoml.exceptions import YataiDeploymentException


def get_deployment_operator(deployment_pb):
    operator = deployment_pb.spec.operator

    if operator == DeploymentSpec.AWS_SAGEMAKER:
        from bentoml.deployment.sagemaker import SageMakerDeploymentOperator

        return SageMakerDeploymentOperator()
    elif operator == DeploymentSpec.AWS_LAMBDA:
        from bentoml.deployment.aws_lambda import AwsLambdaDeploymentOperator

        return AwsLambdaDeploymentOperator()
    elif operator == DeploymentSpec.KUBERNETES:
        raise NotImplementedError("Kubernetes deployment operator is not implemented")
    elif operator == DeploymentSpec.CUSTOM:
        raise NotImplementedError("Custom deployment operator is not implemented")
    else:
        raise YataiDeploymentException("DeployOperator must be set")


class DeploymentOperatorBase(object):

    __metaclass__ = ABCMeta

    @abstractmethod
    def apply(self, deployment_pb, yatai_service, prev_deployment):
        """
        Create deployment based on deployment_pb spec - the bento name and version
        must be found in the given BentoRepository
        """

    @abstractmethod
    def delete(self, deployment_pb, yatai_service):
        """
        Delete deployment based on deployment_pb spec - the bento name and version
        must be found in the given BentoRepository
        """

    @abstractmethod
    def describe(self, deployment_pb, yatai_service):
        """
        Fetch status on an existing deployment created with deployment_pb spec - the
        bento name and version must be found in the given BentoRepository
        """
