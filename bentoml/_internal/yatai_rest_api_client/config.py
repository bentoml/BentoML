import os
import logging
from typing import List
from typing import Optional
from pathlib import Path

import attr
import yaml
import cattr

from bentoml.exceptions import YataiRESTApiClientError

from .yatai import YataiRESTApiClient
from ..configuration.containers import BENTOML_HOME

logger = logging.getLogger(__name__)

default_context_name = "default"


def get_config_path() -> Path:
    return Path(BENTOML_HOME) / ".yatai.yaml"


@attr.define
class YataiClientContext:
    name: str
    endpoint: str
    api_token: str
    email: Optional[str] = attr.field(default=None)

    def get_yatai_rest_api_client(self) -> YataiRESTApiClient:
        return YataiRESTApiClient(self.endpoint, self.api_token)

    def get_email(self) -> str:
        if not self.email:
            cli = self.get_yatai_rest_api_client()
            user = cli.get_current_user()
            if user is None:
                raise YataiRESTApiClientError(
                    "Unable to get current user from yatai server"
                )
            self.email = user.email
            add_context(self, ignore_warning=True)
        return self.email


@attr.define
class YataiClientConfig:
    contexts: List[YataiClientContext] = attr.field(factory=list)
    current_context_name: str = attr.field(default=default_context_name)

    def get_current_context(self) -> YataiClientContext:
        for ctx in self.contexts:
            if ctx.name == self.current_context_name:
                return ctx
        raise YataiRESTApiClientError(
            f"Not found {self.current_context_name} yatai context, please login!"
        )


_config: YataiClientConfig = YataiClientConfig()


def store_config(config: YataiClientConfig) -> None:
    with open(get_config_path(), "w") as f:
        dct = cattr.unstructure(config)
        yaml.dump(dct, stream=f)


def init_config() -> YataiClientConfig:
    config = YataiClientConfig(contexts=[], current_context_name=default_context_name)
    store_config(config)
    return config


def get_config() -> YataiClientConfig:
    if not os.path.exists(get_config_path()):
        return init_config()
    with open(get_config_path(), "r") as f:
        dct = yaml.safe_load(f)
        if not dct:
            return init_config()
        return cattr.structure(dct, YataiClientConfig)


def add_context(context: YataiClientContext, *, ignore_warning: bool = False) -> None:
    config = get_config()
    for idx, ctx in enumerate(config.contexts):
        if ctx.name == context.name:
            if not ignore_warning:
                logger.warning("Overriding existing Yatai context config: %s", ctx.name)
            config.contexts[idx] = context
            break
    else:
        config.contexts.append(context)
    store_config(config)


def get_current_context() -> YataiClientContext:
    config = get_config()
    return config.get_current_context()


def get_current_yatai_rest_api_client() -> YataiRESTApiClient:
    ctx = get_current_context()
    return ctx.get_yatai_rest_api_client()
