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

from bentoml.proto.deployment_pb2 import DeployPlatform
from bentoml.exceptions import BentoMLDeploymentException


class DeploymentService(object):
    def apply(self, deployment):
        platform = deployment.spec.platform

        if platform == DeployPlatform.AWS_LAMBDA:
            pass
        elif platform == DeployPlatform.AWS_SAGEMAKER:
            pass
        elif platform == DeployPlatform.GCP_FUNCTION:
            pass
        elif platform == DeployPlatform.KUBERNETES:
            pass
        else:
            raise BentoMLDeploymentException(
                "Platform '{}' not supported".format(platform)
            )

        # save_deployment_config(deployment)

    def delete(self, deployment_name):
        pass

    def get(self, deployment_name):
        pass

    def describe(self, deployment_name):
        pass

    def list(self, filter=None, labels=None, offset=0, limit=100):
        pass
