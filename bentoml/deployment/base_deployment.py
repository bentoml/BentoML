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

from enum import Enum

from bentoml.archive import load


class LegacyDeployment(object):
    """LegacyDeployment is spec for describing what actions deployment should have
    to interact with BentoML cli and BentoML service archive.
    """

    def __init__(self, archive_path):
        self.bento_service = load(archive_path)
        self.archive_path = archive_path

    def deploy(self):
        """Deploy bentoml service.

        """
        raise NotImplementedError

    def check_status(self):
        """Check deployment status

        """
        raise NotImplementedError

    def delete(self):
        """Delete deployment, if deployment is active

        """
        raise NotImplementedError


class DeploymentService(object):
    @staticmethod
    def create(
        deployment_name, bento_service_name, bento_service_version, platform, config
    ):
        pass

    @staticmethod
    def apply(deployment_name, bento_service_name, bento_service_version, config=None):
        pass

    @staticmethod
    def delete(deployment_name):
        pass

    @staticmethod
    def get(deployment_name):
        pass

    @staticmethod
    def describe(deployment_name):
        pass


class StatusCode(Enum):
    OK = 0
    CANCELLED = 1
    UNKNOWN = 2
    INVALID_ARGUMENT = 3
    NOT_FOUND = 4
    ALREADY_EXISTS = 5
    PERMISSION_DENNIED = 6
    UNAUTHENTICATED = 7
    RESOURCE_EXHAUSTED = 8
    FAILED_PRECONDITION = 9
    ABORTED = 10
    INTERNAL = 11
    UNAVAILABLE = 12


class ResponseStatus:
    def __init__(self, status_code=StatusCode.OK, error_message=None):
        self.status_code = status_code
        self.error_message = error_message


class CreateDeploymentResponse:
    pass


class ApplyDeploymentResponse:
    pass


class DeleteDeploymentResponse:
    pass


class GetDeploymentResponse:
    pass


class DescribeDeploymentResponse:
    pass
