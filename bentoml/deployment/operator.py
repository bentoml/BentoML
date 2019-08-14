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

from six import add_metaclass
from abc import abstractmethod, ABCMeta

from bentoml.proto import deployment_pb2
from bentoml.exceptions import BentoMLDeploymentException


def get_deployment_operator(deployment_pb):
    operator = deployment_pb.spec.operator

    if operator == deployment_pb2.AWS_SAGEMAKER:
        from bentoml.deployment.sagemaker import SageMakerDeploymentOperator

        return SageMakerDeploymentOperator()
    elif operator == deployment_pb2.AWS_LAMBDA:
        pass
    elif operator == deployment_pb2.GCP_FUNCTION:
        raise NotImplementedError("GCP function deployment operator is not implemented")
    elif operator == deployment_pb2.KUBERNETES:
        raise NotImplementedError("Kubernetes deployment operator is not implemented")
    elif operator == deployment_pb2.CUSTOM:
        raise NotImplementedError("Custom deployment operator is not implemented")
    else:
        raise BentoMLDeploymentException("DeployOperator must be set")


@add_metaclass(ABCMeta)
class DeploymentOperatorBase(object):
    def __init__(self, operator_config=None):
        self.config = operator_config

    @abstractmethod
    def apply(self, deployment_pb):
        pass

    @abstractmethod
    def delete(self, deployment_pb):
        pass

    @abstractmethod
    def describe(self, deployment_pb):
        pass
