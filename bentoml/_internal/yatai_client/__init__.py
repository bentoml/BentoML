# import logging
# from typing import TYPE_CHECKING, Optional
#
# from ..utils import cached_property
# from .bento_repository_api import BentoRepositoryAPIClient
#
# if TYPE_CHECKING:
#     from .proto.yatai_service_pb2_grpc import YataiStub
#
# logger = logging.getLogger(__name__)


import logging

from bentoml._internal.utils import cached_property
from bentoml._internal.yatai_client.bento_repository_api import BentoRepositoryAPIClient
from bentoml._internal.yatai_client.deployment_api import DeploymentAPIClient

logger = logging.getLogger(__name__)


class YataiClient:
    """
    Python Client for interacting with YataiService
    """

    def __init__(self, yatai_server_name: str):
        self._yatai_service = get_yatai_service()
        self.bundle_api_client = None
        self.deploy_api_client = None

    @cached_property
    def bundles(self):
        return BentoRepositoryAPIClient(self._yatai_service)

    @cached_property
    def deployment(self):
        return DeploymentAPIClient(self._yatai_service)

    # def __init__(self, yatai_service: Optional["YataiStub"] = None):
    #     self.yatai_service = yatai_service if yatai_service else get_yatai_service()
    #     self.bento_repository_api_client = None
    #     self.deployment_api_client = None
    #
    # @cached_property
    # def repository(self) -> "BentoRepositoryAPIClient":
    #     return BentoRepositoryAPIClient(self.yatai_service)


def get_yatai_client(yatai_url: str = None) -> "YataiClient":
    """
    Args:
        yatai_url (`str`):
            Yatai Service URL address.

    Returns:
        :obj:`~YataiClient`, a python client
        to interact with :obj:`Yatai` gRPC server.

    Example::

        from bentoml.yatai.client import get_yatai_client

        custom_url = 'https://remote.yatai:50050'
        yatai_client = get_yatai_client(custom_url)
    """

    pass
    # yatai_service = get_yatai_service(channel_address=yatai_url)
    # return YataiClient(yatai_service=yatai_service)


@inject
def get_yatai_service(
    channel_address: str,
    access_token: str,
    access_token_header: str,
    tls_root_ca_cert: str,
    tls_client_key: str,
    tls_client_cert: str,
):
    import grpc

    from bentoml._internal.yatai_client.interceptor import header_client_interceptor
    from bentoml.yatai_client.proto.yatai_service_pb2_grpc import YataiStub

    channel_address = channel_address.strip()
    schema, addr = parse_grpc_url(channel_address)
    header_adder_interceptor = header_client_interceptor.header_adder_interceptor(
        access_token_header, access_token
    )
    if schema in ("grpc", "https"):
        tls_root_ca_cert = tls_root_ca_cert or certifi.where()
        with open(tls_client_cert, "rb") as fb:
            ca_cert = fb.read()
        if tls_client_key:
            with open(tls_client_key, "rb") as fb:
                tls_client_key = fb.read()
        if tls_client_cert:
            with open(tls_client_cert, "rb") as fb:
                tls_client_cert = fb.read()
        credentials = grpc.ssl_channel_credentials(
            root_certificates=ca_cert,
            private_key=tls_client_key,
            certificate_chain=tls_client_cert,
        )
        channel = grpc.secure_channel(addr, credentials)
    else:
        channel = grpc.insecure_channel(addr)

    return YataiStub(grpc.intercept_channel(channel, header_adder_interceptor))
