import logging
import multiprocessing
import os
from typing import TYPE_CHECKING

from deepmerge import always_merger
from schema import And, Optional, Or, Schema, SchemaError, Use
from simple_di import Provide, Provider, container, providers

from bentoml import __version__
from bentoml._internal.configuration import expand_env_var, get_bentoml_deploy_version
from bentoml._internal.utils import get_free_port
from bentoml._internal.utils.ruamel_yaml import YAML
from bentoml.exceptions import BentoMLConfigException

if TYPE_CHECKING:
    from ..marshal.marshal import MarshalApp

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
        "bento_bundle": {
            "deployment_version": Or(str, None),
            "default_docker_base_image": Or(str, None),
        },
        "bundle_store_path": Or(str, None),
        "bento_server": {
            "port": And(int, lambda port: port > 0),
            "workers": Or(And(int, lambda workers: workers > 0), None),
            "timeout": And(int, lambda timeout: timeout > 0),
            "max_request_size": And(int, lambda size: size > 0),
            "microbatch": {
                Optional("enabled", default=True): bool,
                "workers": Or(And(int, lambda workers: workers > 0), None),
                "max_batch_size": Or(And(int, lambda size: size > 0), None),
                "max_latency": Or(And(int, lambda latency: latency > 0), None),
            },
            "ngrok": {"enabled": bool},
            "swagger": {"enabled": bool},
            "metrics": {"enabled": bool, "namespace": str},
            "feedback": {"enabled": bool},
            "logging": {"level": str},
            "cors": {
                "enabled": bool,
                "access_control_allow_origin": Or(str, None),
                "access_control_allow_credentials": Or(bool, None),
                "access_control_allow_headers": Or([str], str, None),
                "access_control_allow_methods": Or([str], str, None),
                "access_control_max_age": Or(int, None),
                "access_control_expose_headers": Or([str], str, None),
            },
        },
        "logging": {
            "level": And(
                str,
                lambda level: level.isupper(),
                error="logging.level must be all upper case letters",
            ),
            "console": {"enabled": bool},
            "file": {"enabled": bool, "directory": Or(str, None)},
            "advanced": {"enabled": bool, "config": Or(dict, None)},
        },
        "tracing": {
            "type": Or(
                And(str, Use(str.lower), lambda s: s in ("zipkin", "jaeger")), None
            ),
            Optional("zipkin"): {"url": Or(str, None)},
            Optional("jaeger"): {"address": Or(str, None), "port": Or(int, None)},
        },
        "adapters": {"image_input": {"default_extensions": [str]}},
        "yatai": {
            "default_server": Or(str, None),
            "servers": {
                str: {
                    "url": Or(str, None),
                    "access_token": Or(str, None),
                    "access_token_header": Or(str, None),
                    "tls": {
                        "root_ca_cert": Or(str, None),
                        "client_key": Or(str, None),
                        "client_cert": Or(str, None),
                        "client_certificate_file": Or(str, None),
                    },
                },
            },
        },
    }
)


class BentoMLConfiguration:
    def __init__(
        self,
        default_config_file: str = None,
        default_yatai_server_config_file: str = None,
        override_config_file: str = None,
        override_yatai_config_file: str = None,
        validate_schema: bool = True,
    ):
        # Default configuration
        if default_config_file is None:
            default_config_file = os.path.join(
                os.path.dirname(__file__), "default_configuration.yml"
            )
        if default_yatai_server_config_file is None:
            default_config_file = os.path.join(
                os.path.dirname(__file__), "default_yatai_configuration.yml"
            )

        with open(default_config_file, "rb") as f:
            self.config = YAML().load(f.read())

        with open(default_config_file, "rb") as f:
            self.config["yatai"] = YAML().load(f.read())

        if validate_schema:
            try:
                SCHEMA.validate(self.config)
            except SchemaError as e:
                raise BentoMLConfigException(
                    "Default configuration 'default_configuration.yml' does not"
                    " conform to the required schema."
                ) from e

        # User override configuration
        override_default_config = (
            True
            if override_config_file is not None
            or override_yatai_config_file is not None
            else False
        )
        if override_config_file is not None:
            LOGGER.info("Applying user config override from %s" % override_config_file)
            if not os.path.exists(override_config_file):
                raise BentoMLConfigException(
                    f"Config file {override_config_file} not found"
                )
            with open(override_config_file, "rb") as f:
                override_config = YAML().load(f.read())
            always_merger.merge(self.config, override_config)

        if override_yatai_config_file is not None:
            LOGGER.info(
                "Applying user yatai server config override from %s"
                % override_yatai_config_file
            )
            if not os.path.exists(override_yatai_config_file):
                raise BentoMLConfigException(
                    f"Config file {override_yatai_config_file} not found"
                )
            with open(override_yatai_config_file, "rb") as f:
                override_yatai_config = YAML().load(f.read())
            always_merger.merge(self.config, override_yatai_config)

        if validate_schema and override_default_config:
            try:
                SCHEMA.validate(self.config)
            except SchemaError as e:
                raise BentoMLConfigException(
                    "Configuration after user override does not conform to"
                    " the required schema."
                ) from e

    def override(self, keys: list, value):
        if keys is None:
            raise BentoMLConfigException("Configuration override key is None.")
        if len(keys) == 0:
            raise BentoMLConfigException("Configuration override key is empty.")
        if value is None:
            return

        c = self.config
        for key in keys[:-1]:
            if key not in c:
                raise BentoMLConfigException(
                    "Configuration override key is invalid, %s" % keys
                )
            c = c[key]
        c[keys[-1]] = value

        try:
            SCHEMA.validate(self.config)
        except SchemaError as e:
            raise BentoMLConfigException(
                "Configuration after applying override does not conform"
                " to the required schema, key=%s, value=%s." % (keys, value)
            ) from e

    def as_dict(self) -> dict:
        return self.config


@container
class BentoMLContainerClass:

    config = providers.Configuration()

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

    api_server_workers = providers.Factory(
        lambda workers: workers or (multiprocessing.cpu_count() // 2) + 1,
        config.bento_server.workers,
    )

    '''
    bentoml_home = providers.Factory(
        lambda: expand_env_var(
            os.environ.get(
                "BENTOML_HOME", # os.path.join(os.environ["XDG_DATA_HOME"], "bentoml")
            )
        )
    )
    '''
    bentoml_home = providers.Factory(
        lambda: expand_env_var(
            os.environ.get(
                "BENTOML_HOME",
                os.path.join("~", "bentoml"),
            )
        )
    )
    bundle_store_path = providers.Factory(
        lambda default, customized: customized or default,
        providers.Factory(
            os.path.join,
            bentoml_home._provide(),
            "bundles",
        ),
        config.bundle_store_path,
    )

    bundle_path: Provider[str] = providers.Static("")

    service_host: Provider[str] = providers.Static("0.0.0.0")
    service_port: Provider[int] = config.bento_server.port

    forward_host: Provider[str] = providers.Static("localhost")
    forward_port: Provider[int] = providers.SingletonFactory(get_free_port)

    @providers.Factory
    @staticmethod
    def model_server():
        from ..server.gunicorn_model_server import GunicornModelServer

        return GunicornModelServer()

    @providers.Factory
    @staticmethod
    def proxy_server():
        from ..server.gunicorn_marshal_server import GunicornMarshalServer

        return GunicornMarshalServer()

    @providers.Factory
    @staticmethod
    def proxy_app() -> "MarshalApp":
        from bentoml._internal.marshal.marshal import MarshalApp

        return MarshalApp()

    @providers.Factory
    @staticmethod
    def model_app():
        from ..server.model_app import ModelApp

        return ModelApp()

    prometheus_lock = providers.SingletonFactory(multiprocessing.Lock)

    prometheus_multiproc_dir = providers.Factory(
        os.path.join, bentoml_home, "prometheus_multiproc_dir",
    )

    @providers.SingletonFactory
    @staticmethod
    def metrics_client(
        multiproc_lock=prometheus_lock,
        multiproc_dir=prometheus_multiproc_dir,
        namespace=config.bento_server.metrics.namespace,
    ):
        from ..metrics.prometheus import PrometheusClient

        return PrometheusClient(
            multiproc_lock=multiproc_lock,
            multiproc_dir=multiproc_dir,
            namespace=namespace,
        )

    bento_bundle_deployment_version = providers.Factory(
        get_bentoml_deploy_version,
        providers.Factory(
            lambda default, customized: customized or default,
            __version__.split("+")[0],
            config.bento_bundle.deployment_version,
        ),
    )
    default_yatai_server = config.yatai.servers[config.yatai.default_server]

    default_yatai_server_tls_root_ca_cert = providers.Factory(
        lambda current, deprecated: current or deprecated,
        config.yatai.servers[config.yatai.default_server].tls.root_ca_cert,
        config.yatai.servers[config.yatai.default_server].tls.client_certificate_file,
    )

    logging_file_directory = providers.Factory(
        lambda default, customized: customized or default,
        providers.Factory(os.path.join, bentoml_home, "logs",),
        config.logging.file.directory,
    )


BentoMLContainer = BentoMLContainerClass()
