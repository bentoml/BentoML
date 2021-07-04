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

from abc import abstractmethod, ABCMeta


def get_deployment_operator(yatai_service, deployment_pb):  # pylint:disable=W0613
    return DeploymentOperatorBase(yatai_service)


class DeploymentOperatorBase(object):
    def __init__(self, yatai_service):
        self.yatai_service = yatai_service

    __metaclass__ = ABCMeta

    @abstractmethod
    def add(self, deployment_pb):
        """
        Create deployment based on deployment_pb spec - the bento name and version
        must be found in the given BentoRepository
        """

    @abstractmethod
    def update(self, deployment_pb, previous_deployment):
        """
        Update existing deployment based on deployment_pb spec
        """

    @abstractmethod
    def delete(self, deployment_pb):
        """
        Delete deployment based on deployment_pb spec
        """

    @abstractmethod
    def describe(self, deployment_pb):
        """
        Fetch current state of an existing deployment created with deployment_pb spec
        """
