import logging
import multiprocessing
import os
from typing import TYPE_CHECKING

from cerberus import Validator
from deepmerge import always_merger
from simple_di import Provide, Provider, container, providers

# TODO separate out yatai version. Should we go with the BentoML version or start new?
from bentoml import __version__
from yatai.configuration import expand_env_var
from yatai.exceptions import YataiConfigurationException
from yatai.utils.ruamel_yaml import YAML


LOGGER = logging.getLogger(__name__)

YATAI_REPOSITORY_S3 = "s3"
YATAI_REPOSITORY_GCS = "gcs"
YATAI_REPOSITORY_FILE_SYSTEM = "file_system"
YATAI_REPOSITORY_TYPES = [
    YATAI_REPOSITORY_FILE_SYSTEM,
    YATAI_REPOSITORY_S3,
    YATAI_REPOSITORY_GCS,
]


yatai_configuration_schema = {
    "servers": {
        "type": "list",
        "schema": {
            "type": "dict",
            "schema": {
                "name": {"type": "string", "default": "default"},
                "is_default": {"type": "boolean", "default": True},
                "url": {"type": "string"},
                "access_token": {"type": "string"},
                "access_token_header": {"type": "string", "default": "access_token"},
                "tls": {
                    "type": "dict",
                    "schema": {
                        "root_ca_cert": {"type": "string"},
                        "client_certificate_file": {"type": "string"},
                        "client_key": {"type": "string"},
                        "client_cert": {"type": "string"},
                    },
                },
            },
        },
    },
    "repository": {
        "type": "dict",
        "schema": {
            "type": {
                "allowed": YATAI_REPOSITORY_TYPES,
                "default": YATAI_REPOSITORY_FILE_SYSTEM,
            },
            "file_system": {"type": "dict", "allow_unknown": True},
            "s3": {
                "type": "dict",
                "schema": {
                    "url": {"type": "string"},
                    "endpoint_url": {"type": "string"},
                    "signature_version": {"type": "string"},
                    "expiration": {"type": "integer"},
                },
            },
            "gcs": {
                "type": "dict",
                "schema": {
                    "url": {"type": "string"},
                    "expiration": {"type": "integer"},
                },
            },
        },
    },
    "database": {"type": "dict", "schema": {"url": {"type": "string"}},},
    "namespace": {"type": "string",},
    "logging": {"type": "dict", "schema": {"path": {"type": "string"}},},
}



class YataiConfiguration:
    def __init__(
        self,
        default_config_file: str = None,
        override_config_file: str = None,
        validate_schema: bool = True,
    ):
        # Default configuration
        if default_config_file is None:
            default_config_file = os.path.join(
                os.path.dirname(__file__), "default_configuration.yml"
            )

        with open(default_config_file, "rb") as f:
            self.config = YAML().load(f.read())

        if validate_schema:
            yatai_config_validator = Validator(yatai_configuration_schema)
            validation_result = yatai_config_validator.validate(self.config)
            if not validation_result:
                raise YataiConfigurationException(yatai_config_validator.errors)


        # User override configuration
        if override_config_file is not None:
            LOGGER.info("Applying user config override from %s" % override_config_file)
            if not os.path.exists(override_config_file):
                raise YataiConfigurationException(
                    f"Config file {override_config_file} not found"
                )

            with open(override_config_file, "rb") as f:
                override_config = YAML().load(f.read())
            always_merger.merge(self.config, override_config)

            if validate_schema:
                yatai_config_validator = Validator(yatai_configuration_schema)
                validation_result = yatai_config_validator.validate(self.config)
                if not validation_result:
                    raise YataiConfigurationException(yatai_config_validator.errors)

    def override(self, keys: list, value):
        if keys is None:
            raise YataiConfigurationException("Configuration override key is None.")
        if len(keys) == 0:
            raise YataiConfigurationException("Configuration override key is empty.")
        if value is None:
            return

        c = self.config
        for key in keys[:-1]:
            if key not in c:
                raise YataiConfigurationException(
                    "Configuration override key is invalid, %s" % keys
                )
            c = c[key]
        c[keys[-1]] = value

        try:
            SCHEMA.validate(self.config)
        except SchemaError as e:
            raise YataiConfigurationException(
                "Configuration after applying override does not conform"
                " to the required schema, key=%s, value=%s." % (keys, value)
            ) from e

    def as_dict(self) -> dict:
        return self.config


@container
class YataiContainerClass:

    config = providers.Configuration()

    bentoml_home = providers.Factory(
        lambda: expand_env_var(
            os.environ.get("BENTOML_HOME", os.path.join("~", "bentoml"))
        )
    )

    @providers.SingletonFactory
    @staticmethod
    def tracer(
        tracer_type: str = Provide[config.tracing.type],
        zipkin_server_url: str = Provide[config.tracing.zipkin.url],
        jaeger_server_address: str = Provide[config.tracing.jaeger.address],
        jaeger_server_port: int = Provide[config.tracing.jaeger.port],
    ):
        if tracer_type and tracer_type.lower() == "zipkin" and zipkin_server_url:
            from ..tracing.zipkin import get_zipkin_tracer

            return get_zipkin_tracer(zipkin_server_url)
        elif (
            tracer_type
            and tracer_type.lower() == "jaeger"
            and jaeger_server_address
            and jaeger_server_port
        ):
            from ..tracing.jaeger import get_jaeger_tracer

            return get_jaeger_tracer(jaeger_server_address, jaeger_server_port)
        else:
            from ..tracing.noop import NoopTracer

            return NoopTracer()

    @providers.SingletonFactory
    @staticmethod
    def access_control_options(
        allow_credentials=config.bento_server.cors.access_control_allow_credentials,
        expose_headers=config.bento_server.cors.access_control_expose_headers,
        allow_methods=config.bento_server.cors.access_control_allow_methods,
        allow_headers=config.bento_server.cors.access_control_allow_headers,
        max_age=config.bento_server.cors.access_control_max_age,
    ):
        import aiohttp_cors

        kwargs = dict(
            allow_credentials=allow_credentials,
            expose_headers=expose_headers,
            allow_methods=allow_methods,
            allow_headers=allow_headers,
            max_age=max_age,
        )

        filtered_kwargs = {k: v for k, v in kwargs.items() if v is not None}

        return aiohttp_cors.ResourceOptions(**filtered_kwargs)

    @providers.SingletonFactory
    @staticmethod
    def metrics_client():
        from ..metrics.prometheus import PrometheusClient

        return PrometheusClient(multiproc=False, namespace="YATAI")

    database_url = providers.Factory(
        lambda default, customized: customized or default,
        providers.Factory(
            "sqlite:///{}".format,
            providers.Factory(os.path.join, bentoml_home, "storage.db"),
        ),
        config.yatai.database.url,
    )

    file_system_directory = providers.Factory(
        lambda default, customized: customized or default,
        providers.Factory(os.path.join, bentoml_home, "repository"),
        config.yatai.repository.file_system.directory,
    )

    # TODO add shortcut to default yatai remote address
    default_server = providers.Factory(
        lambda default, customized: customized or default, "option 1", "default",
    )

    tls_root_ca_cert = providers.Factory(
        lambda current, deprecated: current or deprecated,
        default_server.tls.root_ca_cert,
        default_server.tls.client_certificate_file,
    )

    logging_file_directory = providers.Factory(
        lambda default, customized: customized or default,
        providers.Factory(os.path.join, bentoml_home, "logs",),
        config.logging.file.directory,
    )

    logging_path = providers.Factory(
        lambda default, customized: customized or default,
        providers.Factory(os.path.join, logging_file_directory, "yatai_web_server.log"),
        config.yatai.logging.path,
    )


YataiContainer = YataiContainerClass()
