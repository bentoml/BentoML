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

# List of APIs for accessing remote or local yatai service via Python


import logging

from bentoml.yatai import get_yatai_service
from bentoml.yatai.client.bento_repository_api import BentoRepositoryAPIClient
from bentoml.yatai.client.deployment_api import DeploymentAPIClient


logger = logging.getLogger(__name__)


class YataiClient:
    """Python Client for interacting with YataiService
    """

    def __init__(self, yatai_service=None):
        self.yatai_service = yatai_service if yatai_service else get_yatai_service()
        self.bento_repository_api_client = None
        self.deployment_api_client = None

    @property
    def repository(self):
        if not self.bento_repository_api_client:
            self.bento_repository_api_client = BentoRepositoryAPIClient(
                self.yatai_service
            )

        return self.bento_repository_api_client

    @property
    def deployment(self):
        if not self.deployment_api_client:
            self.deployment_api_client = DeploymentAPIClient(self.yatai_service)

        return self.deployment_api_client
