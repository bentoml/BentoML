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

from bentoml.yatai.yatai_service import get_yatai_service
from bentoml.yatai.client.bento_repository_api import BentoRepositoryAPIClient
from bentoml.yatai.client.deployment_api import DeploymentAPIClient

from bentoml.utils import cached_property

logger = logging.getLogger(__name__)


class YataiClient:
    """Python Client for interacting with YataiService
    """

    def __init__(self, yatai_service=None):
        self.yatai_service = yatai_service if yatai_service else get_yatai_service()
        self.bento_repository_api_client = None
        self.deployment_api_client = None

    @cached_property
    def repository(self):
        return BentoRepositoryAPIClient(self.yatai_service)

    @cached_property
    def deployment(self):
        return DeploymentAPIClient(self.yatai_service)


def get_yatai_client(yatai_service_channel_address=None):
    """
    Args:
        yatai_service_channel_address: String. Yatai Service URL address.

    Returns:
        YataiClient instance

    Example:

    >>>  from bentoml.yatai.client import get_yatai_client
    >>>
    >>>  remote_yatai_service_address = 'https://remote.yatai:50050'
    >>>  yatai_client = get_yatai_client(remote_yatai_service_address)
    >>>
    >>>  local_yatai_client = get_yatai_client()
    """
    yatai_service = get_yatai_service(channel_address=yatai_service_channel_address)
    return YataiClient(yatai_service=yatai_service)
