import logging
import multiprocessing
import os
import typing as t
from typing import TYPE_CHECKING

import yaml
from deepmerge import always_merger
from schema import And, Optional, Or, Schema, SchemaError, Use
from simple_di import Provide, Provider, container, providers

from ...exceptions import BentoMLConfigException
from ..utils import get_free_port
from . import expand_env_var

if TYPE_CHECKING:
    from pyarrow._plasma import PlasmaClient

    from ..server.metrics.prometheus import PrometheusClient


LOGGER = logging.getLogger(__name__)
SYSTEM_HOME = os.path.expanduser("~")


SCHEMA = Schema(
    {
        "bento_server": {
            "port": And(int, lambda port: port > 0),
            "workers": Or(And(int, lambda workers: workers > 0), None),
            "timeout": And(int, lambda timeout: timeout > 0),
            "max_request_size": And(int, lambda size: size > 0),
            "batch_options": {
                Optional("enabled", default=True): bool,
                "max_batch_size": Or(And(int, lambda size: size > 0), None),
                "max_latency_ms": Or(And(int, lambda latency: latency > 0), None),
            },
            "ngrok": {"enabled": bool},
            "metrics": {"enabled": bool, "namespace": str},
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
        Optional("yatai"): {
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
        override_config_file: t.Optional[str] = None,
        validate_schema: bool = True,
    ):
        # Load default configuration
        default_config_file = os.path.join(
            os.path.dirname(__file__), "default_configuration.yaml"
        )
        with open(default_config_file, "rb") as f:
            self.config = yaml.safe_load(f)

        if validate_schema:
            try:
                SCHEMA.validate(self.config)
            except SchemaError as e:
                raise BentoMLConfigException(
                    "Default configuration 'default_configuration.yml' does not"
                    " conform to the required schema."
                ) from e

        # User override configuration
        if override_config_file is not None:
            LOGGER.info("Applying user config override from %s" % override_config_file)
            if not os.path.exists(override_config_file):
                raise BentoMLConfigException(
                    f"Config file {override_config_file} not found"
                )
            with open(override_config_file, "rb") as f:
                override_config = yaml.safe_load(f)
            always_merger.merge(self.config, override_config)

            if validate_schema:
                try:
                    SCHEMA.validate(self.config)
                except SchemaError as e:
                    raise BentoMLConfigException(
                        "Configuration after user override does not conform to"
                        " the required schema."
                    ) from e

        # TODO:
        # Find local yatai configurations and merge with it

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

    bentoml_home = providers.Static(
        expand_env_var(
            os.environ.get("BENTOML_HOME", os.path.join(SYSTEM_HOME, "bentoml"))
        )
    )

    default_bento_store_base_dir: Provider[str] = providers.Factory(
        os.path.join,
        bentoml_home,
        "bentos",
    )

    default_model_store_base_dir: Provider[str] = providers.Factory(
        os.path.join,
        bentoml_home,
        "models",
    )

    @providers.SingletonFactory
    @staticmethod
    def bento_store(base_dir=default_bento_store_base_dir):
        from ..bento import BentoStore

        return BentoStore(base_dir)

    @providers.SingletonFactory
    @staticmethod
    def model_store(base_dir=default_model_store_base_dir):
        from ..models.store import ModelStore

        return ModelStore(base_dir)

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

    logging_file_directory = providers.Factory(
        lambda default, customized: customized or default,
        providers.Factory(
            os.path.join,
            bentoml_home,
            "logs",
        ),
        config.logging.file.directory,
    )


BentoMLContainer = BentoMLContainerClass()


@container
class BentoServerContainerClass:

    bentoml_container = BentoMLContainer
    config = bentoml_container.config.bento_server

    @providers.SingletonFactory
    @staticmethod
    def access_control_options(
        allow_origins: t.List[str] = Provide[config.cors.access_control_allow_origin],
        allow_credentials: t.List[str] = Provide[
            config.cors.access_control_allow_credentials
        ],
        expose_headers: t.List[str] = Provide[
            config.cors.access_control_expose_headers
        ],
        allow_methods: t.List[str] = Provide[config.cors.access_control_allow_methods],
        allow_headers: t.List[str] = Provide[config.cors.access_control_allow_headers],
        max_age: int = Provide[config.cors.access_control_max_age],
    ) -> t.Dict:
        kwargs = dict(
            allow_origins=allow_origins,
            allow_credentials=allow_credentials,
            expose_headers=expose_headers,
            allow_methods=allow_methods,
            allow_headers=allow_headers,
            max_age=max_age,
        )

        filtered_kwargs = {k: v for k, v in kwargs.items() if v is not None}
        return filtered_kwargs

    api_server_workers = providers.Factory(
        lambda workers: workers or (multiprocessing.cpu_count() // 2) + 1,
        config.workers,
    )

    service_host: Provider[str] = providers.Static("0.0.0.0")
    service_port: Provider[int] = config.port

    forward_host: Provider[str] = providers.Static("localhost")
    forward_port: Provider[int] = providers.SingletonFactory(get_free_port)

    prometheus_lock = providers.SingletonFactory(multiprocessing.Lock)

    prometheus_multiproc_dir = providers.Factory(
        os.path.join,
        bentoml_container.bentoml_home,
        "prometheus_multiproc_dir",
    )

    @providers.SingletonFactory
    @staticmethod
    def metrics_client(
        multiproc_dir: str = Provide[prometheus_multiproc_dir],
        namespace: str = Provide[config.metrics.namespace],
    ) -> "PrometheusClient":
        from ..server.metrics.prometheus import PrometheusClient

        return PrometheusClient(
            multiproc_dir=multiproc_dir,
            namespace=namespace,
        )

    # Mapping from runner name to RunnerApp file descriptor
    remote_runner_mapping: Provider[t.Dict[str, int]] = providers.Static(dict())
    plasma_db: "PlasmaClient" = providers.Static(None)


BentoServerContainer = BentoServerContainerClass()
