import logging
import os

from deepmerge import always_merger
from schema import And, Or, Schema, SchemaError, Use
from simple_di import container, providers

# TODO separate out yatai version. Should we go with the BentoML version or start new?
from yatai import __version__
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


SCHEMA = Schema(
    {
        "repository": {
            "type": And(
                str,
                lambda type: type in YATAI_REPOSITORY_TYPES,
                error="yatai.repository.type must be one of %s"
                % YATAI_REPOSITORY_TYPES,
            ),
            "file_system": {"directory": Or(str, None)},
            "s3": {
                "url": Or(str, None),
                "endpoint_url": Or(str, None),
                "signature_version": Or(str, None),
                "expiration": Or(int, None),
            },
            "gcs": {"url": Or(str, None), "expiration": Or(int, None)},
        },
        "database": {"url": Or(str, None)},
        "namespace": str,
        "logging": {"path": Or(str, None)},
    }
)


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
            try:
                SCHEMA.validate(self.config)
            except SchemaError as e:
                raise YataiConfigurationException(
                    "Default configuration 'default_configuration.yml' does not"
                    " conform to the required schema."
                ) from e

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
                try:
                    SCHEMA.validate(self.config)
                except SchemaError as e:
                    raise YataiConfigurationException(
                        "Configuration after user override does not conform to"
                        " the required schema."
                    ) from e

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
    def metrics_client():
        from bentoml._internal.metrics.prometheus import PrometheusClient

        return PrometheusClient(multiproc=False, namespace="YATAI")

    database_url = providers.Factory(
        lambda default, customized: customized or default,
        providers.Factory(
            "sqlite:///{}".format,
            providers.Factory(os.path.join, bentoml_home, "storage.db"),
        ),
        config.database.url,
    )

    file_system_directory = providers.Factory(
        lambda default, customized: customized or default,
        providers.Factory(os.path.join, bentoml_home, "repository"),
        config.repository.file_system.directory,
    )

    logging_file_directory = providers.Factory(
        lambda default, customized: customized or default,
        providers.Factory(os.path.join, bentoml_home, "logs",),
        config.logging.file.directory,
    )

    logging_path = providers.Factory(
        lambda default, customized: customized or default,
        providers.Factory(os.path.join, logging_file_directory, "yatai_web_server.log"),
        config.logging.path,
    )


YataiContainer = YataiContainerClass()
