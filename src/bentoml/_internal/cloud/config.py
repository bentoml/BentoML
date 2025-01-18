from __future__ import annotations

import logging
import typing as t
from http import HTTPStatus

import attr
import yaml
from simple_di import Provide
from simple_di import inject

from ...exceptions import CloudRESTApiClientError
from ..configuration.containers import BentoMLContainer
from ..utils.cattr import bentoml_cattr
from .client import RestApiClient

logger = logging.getLogger(__name__)

default_context_name = "default"

if t.TYPE_CHECKING:
    from pathlib import Path


@attr.define
class CloudClientContext:
    name: str
    endpoint: str
    api_token: str
    email: t.Optional[str] = attr.field(default=None)

    def get_rest_api_client(self) -> RestApiClient:
        return RestApiClient(self.endpoint, self.api_token)

    def get_email(self) -> str:
        if not self.email:
            cli = self.get_rest_api_client()
            user = cli.v1.get_current_user()
            if user is None:
                raise CloudRESTApiClientError(
                    "Unable to get current user from yatai server"
                )
            self.email = user.email
            self.save(ignore_warning=True)
        return self.email

    def save(self, *, ignore_warning: bool = False) -> None:
        config = CloudClientConfig.get_config()
        for idx, ctx in enumerate(config.contexts):
            if ctx.name == self.name:
                if not ignore_warning:
                    logger.warning(
                        "Overriding existing cloud context config: %s", ctx.name
                    )
                config.contexts[idx] = self
                break
        else:
            config.contexts.append(self)
        config.to_yaml_file()


DEFAULT_ENDPOINT = "https://cloud.bentoml.com"


@attr.define
class CloudClientConfig:
    contexts: t.List[CloudClientContext] = attr.field(factory=list)
    current_context_name: str = attr.field(default=default_context_name)

    def get_context(self, context: t.Optional[str] = None) -> CloudClientContext:
        from os import environ

        if "BENTO_CLOUD_API_KEY" in environ:
            return CloudClientContext(
                name="__env__",
                endpoint=environ.get("BENTO_CLOUD_API_ENDPOINT", DEFAULT_ENDPOINT),
                api_token=environ["BENTO_CLOUD_API_KEY"],
            )
        if context is None:
            context = self.current_context_name
        for ctx in self.contexts:
            if ctx.name == context:
                return ctx
        raise CloudRESTApiClientError(
            f"No cloud context {context} found",
            error_code=HTTPStatus.UNAUTHORIZED,
        )

    def set_current_context(self, context: str | None) -> CloudClientContext:
        """Set current context to a new context default."""
        try:
            new_context = self.get_context(context)
        except CloudRESTApiClientError as err:
            raise err from None
        new_config = attr.evolve(self, current_context_name=new_context.name)
        new_config.to_yaml_file()
        return new_context

    @inject
    def to_yaml_file(
        self, cloud_config: Path = Provide[BentoMLContainer.cloud_config]
    ) -> None:
        """Converts this config to yaml file at given path object.
        Note that this method will not return the config object.
        """
        with cloud_config.open("w") as f:
            yaml.dump(bentoml_cattr.unstructure(self), stream=f)

    @classmethod
    def default_config(cls) -> CloudClientConfig:
        """Iinitialize a default config and write it to default yaml path."""
        config = cls(contexts=[], current_context_name=default_context_name)
        config.to_yaml_file()
        return config

    @classmethod
    @inject
    def get_config(
        cls, cloud_config: Path = Provide[BentoMLContainer.cloud_config]
    ) -> CloudClientConfig:
        if not cloud_config.exists():
            return cls.default_config()
        with cloud_config.open("r") as stream:
            try:
                yaml_content = yaml.safe_load(stream)
            except yaml.YAMLError as exc:
                logger.error(exc)
                raise

            if not yaml_content:
                return cls.default_config()
            return bentoml_cattr.structure(yaml_content, cls)


def get_rest_api_client(context: str | None = None) -> RestApiClient:
    cfg = CloudClientConfig.get_config()
    ctx = cfg.get_context(context)
    return ctx.get_rest_api_client()
