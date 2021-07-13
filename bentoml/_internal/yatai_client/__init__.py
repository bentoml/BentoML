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
