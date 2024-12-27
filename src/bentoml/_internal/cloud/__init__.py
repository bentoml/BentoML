from __future__ import annotations

import attrs

from bentoml._internal.cloud.client import RestApiClient

from .base import Spinner
from .bento import BentoAPI
from .config import DEFAULT_ENDPOINT
from .config import CloudClientConfig
from .deployment import DeploymentAPI
from .model import ModelAPI
from .secret import SecretAPI
from .yatai import YataiClient as YataiClient


@attrs.frozen
class BentoCloudClient:
    """
    BentoCloudClient is a client for the BentoCloud API.

    Args:
        api_key: The API key to use for the client. env: BENTO_CLOUD_API_KEY
        endpoint: The endpoint to use for the client. env: BENTO_CLOUD_ENDPOINT
        timeout: The timeout to use for the client. Defaults to 60 seconds.

    Attributes:
        bento: Bento API
        model: Model API
        deployment: Deployment API
        secret: Secret API
    """

    bento: BentoAPI
    model: ModelAPI
    deployment: DeploymentAPI
    secret: SecretAPI

    def __init__(
        self,
        api_key: str | None = None,
        endpoint: str = DEFAULT_ENDPOINT,
        timeout: int = 60,
    ) -> None:
        if api_key is None:
            from ..configuration.containers import BentoMLContainer

            cfg = CloudClientConfig.get_config()
            ctx = cfg.get_context(BentoMLContainer.cloud_context.get())
            api_key = ctx.api_token
            endpoint = ctx.endpoint

        client = RestApiClient(endpoint, api_key, timeout)
        spinner = Spinner()
        bento = BentoAPI(client, spinner=spinner)
        model = ModelAPI(client, spinner=spinner)
        deployment = DeploymentAPI(client)
        secret = SecretAPI(client)

        self.__attrs_init__(
            bento=bento, model=model, deployment=deployment, secret=secret
        )

    @classmethod
    def for_context(cls, context: str | None = None) -> "BentoCloudClient":
        cfg = CloudClientConfig.get_config()
        ctx = cfg.get_context(context)
        return cls(ctx.api_token, ctx.endpoint)
