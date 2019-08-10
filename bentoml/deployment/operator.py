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

from bentoml.proto.deployment_pb2 import DeploymentOperator
from bentoml.exceptions import BentoMLDeploymentException


def get_deployment_operator(deployment_pb):
    operator = deployment_pb.spec.operator

    if operator == DeploymentOperator.AWS_SAGEMAKER:
        from bentoml.deployment.sagemaker import SageMakerDeploymentOperator

        return SageMakerDeploymentOperator()
    elif operator == DeploymentOperator.AWS_LAMBDA:
        pass
    elif operator == DeploymentOperator.GCP_FUNCTION:
        pass
    elif operator == DeploymentOperator.KUBERNETES:
        pass
    elif operator == DeploymentOperator.CUSTOM:
        pass
    else:
        raise BentoMLDeploymentException("DeployOperator must be set")


class DeploymentOperatorBase(object):
    def __init__(self, operator_config=None):
        self.config = operator_config

    def apply(self, deployment_pb):
        raise NotImplementedError

    def delete(self, deployment_pb):
        raise NotImplementedError

    def get(self, deployment_pb):
        raise NotImplementedError
