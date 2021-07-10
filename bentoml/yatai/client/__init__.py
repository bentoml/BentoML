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
from typing import TYPE_CHECKING, Optional

from bentoml.utils import cached_property
from bentoml.yatai.client.bento_repository_api import BentoRepositoryAPIClient
from bentoml.yatai.client.deployment_api import DeploymentAPIClient
from bentoml.yatai.yatai_service import get_yatai_service

if TYPE_CHECKING:
    from bentoml.yatai.proto.yatai_service_pb2_grpc import YataiStub

logger = logging.getLogger(__name__)


class YataiClient:
    """Python Client for interacting with YataiService"""

    def __init__(self, yatai_service: Optional["YataiStub"] = None):
        self.yatai_service = yatai_service if yatai_service else get_yatai_service()
        self.bento_repository_api_client = None
        self.deployment_api_client = None

    @cached_property
    def repository(self) -> "BentoRepositoryAPIClient":
        return BentoRepositoryAPIClient(self.yatai_service)

    @cached_property
    def deployment(self) -> "DeploymentAPIClient":
        return DeploymentAPIClient(self.yatai_service)


def get_yatai_client(yatai_url: str = None) -> "YataiClient":
    """
    Args:
        yatai_url (`str`):
            Yatai Service URL address.

    Returns:
        :obj:`~YataiClient`, a python client to interact with :obj:`Yatai` gRPC server.

    Example::

        from bentoml.yatai.client import get_yatai_client

        custom_url = 'https://remote.yatai:50050'
        yatai_client = get_yatai_client(custom_url)
    """  # noqa: E501

    yatai_service = get_yatai_service(channel_address=yatai_url)
    return YataiClient(yatai_service=yatai_service)
