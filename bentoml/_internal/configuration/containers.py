import os
import uuid
import typing as t
import logging
import multiprocessing
from typing import TYPE_CHECKING
from dataclasses import dataclass

import yaml
from schema import Or
from schema import And
from schema import Use
from schema import Schema
from schema import Optional
from schema import SchemaError
from deepmerge import always_merger
from simple_di import Provide
from simple_di import providers

from . import expand_env_var
from ..utils import validate_or_create_dir
from ...exceptions import BentoMLConfigException

if TYPE_CHECKING:
    from bentoml._internal.models import ModelStore

    from .. import external_typing as ext
    from ..utils.analytics import ServeInfo
    from ..server.metrics.prometheus import PrometheusClient


LOGGER = logging.getLogger(__name__)
SYSTEM_HOME = os.path.expanduser("~")


BENTOML_HOME = expand_env_var(
    str(os.environ.get("BENTOML_HOME", os.path.join(SYSTEM_HOME, "bentoml")))
)
DEFAULT_BENTOS_PATH = os.path.join(BENTOML_HOME, "bentos")
DEFAULT_MODELS_PATH = os.path.join(BENTOML_HOME, "models")


validate_or_create_dir(BENTOML_HOME, DEFAULT_BENTOS_PATH, DEFAULT_MODELS_PATH)

_check_tracing_type: t.Callable[[str], bool] = lambda s: s in ("zipkin", "jaeger")
_larger_than: t.Callable[[int], t.Callable[[int], bool]] = (
    lambda target: lambda val: val > target
)
_larger_than_zero: t.Callable[[int], bool] = _larger_than(0)


def _is_ip_address(addr: str) -> bool:
    import socket

    try:
        socket.inet_aton(addr)
        return True
    except socket.error:
        return False


SCHEMA = Schema(
    {
        "bento_server": {
            "port": And(int, _larger_than_zero),
            "host": And(str, _is_ip_address),
            "backlog": And(int, _larger_than(64)),
            "workers": Or(And(int, _larger_than_zero), None),
            "timeout": And(int, _larger_than_zero),
            "max_request_size": And(int, _larger_than_zero),
            "batch_options": {
                Optional("enabled", default=True): bool,
                "max_batch_size": Or(And(int, _larger_than_zero), None),
                "max_latency_ms": Or(And(int, _larger_than_zero), None),
            },
            "ngrok": {"enabled": bool},
            "metrics": {"enabled": bool, "namespace": str},
            "logging": {
                # TODO add logging level configuration
                "access": {
                    "enabled": bool,
                    "request_content_length": Or(bool, None),
                    "request_content_type": Or(bool, None),
                    "response_content_length": Or(bool, None),
                    "response_content_type": Or(bool, None),
                },
            },
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
        "runners": {
            "logging": {
                # TODO add logging level configuration
                "access": {
                    "enabled": bool,
                    "request_content_length": Or(bool, None),
                    "request_content_type": Or(bool, None),
                    "response_content_length": Or(bool, None),
                    "response_content_type": Or(bool, None),
                },
            },
        },
        "tracing": {
            "type": Or(And(str, Use(str.lower), _check_tracing_type), None),
            "sample_rate": Or(And(float, lambda i: i >= 0 and i <= 1), None),
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
            self.config: t.Dict[str, t.Any] = yaml.safe_load(f)

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

    def override(self, keys: t.List[str], value: t.Any):
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

    def as_dict(self) -> providers.ConfigDictType:
        return t.cast(providers.ConfigDictType, self.config)


@dataclass
class BentoMLContainerClass:

    config = providers.Configuration()

    bentoml_home: str = BENTOML_HOME
    default_bento_store_base_dir: str = DEFAULT_BENTOS_PATH
    default_model_store_base_dir: str = DEFAULT_MODELS_PATH

    @providers.SingletonFactory
    @staticmethod
    def bento_store(base_dir: str = default_bento_store_base_dir):
        from ..bento import BentoStore

        return BentoStore(base_dir)

    @providers.SingletonFactory
    @staticmethod
    def model_store(base_dir: str = default_model_store_base_dir) -> "ModelStore":
        from ..models import ModelStore

        return ModelStore(base_dir)

    @providers.SingletonFactory
    @staticmethod
    def session_id() -> str:
        return uuid.uuid1().hex


BentoMLContainer = BentoMLContainerClass()


@dataclass
class DeploymentContainerClass:
    bentoml_container = BentoMLContainer
    config = bentoml_container.config
    api_server_config = config.bento_server
    runners_config = config.runners

    @providers.SingletonFactory
    @staticmethod
    def serve_info() -> "ServeInfo":
        from ..utils.analytics import get_serve_info

        return get_serve_info()

    @providers.SingletonFactory
    @staticmethod
    def access_control_options(
        allow_origins: t.List[str] = Provide[
            api_server_config.cors.access_control_allow_origin
        ],
        allow_credentials: t.List[str] = Provide[
            api_server_config.cors.access_control_allow_credentials
        ],
        expose_headers: t.List[str] = Provide[
            api_server_config.cors.access_control_expose_headers
        ],
        allow_methods: t.List[str] = Provide[
            api_server_config.cors.access_control_allow_methods
        ],
        allow_headers: t.List[str] = Provide[
            api_server_config.cors.access_control_allow_headers
        ],
        max_age: int = Provide[api_server_config.cors.access_control_max_age],
    ) -> t.Dict[str, t.Union[t.List[str], int]]:
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

    api_server_workers = providers.Factory[int](
        lambda workers: workers or (multiprocessing.cpu_count() // 2) + 1,
        api_server_config.workers,
    )
    service_port = api_server_config.port
    service_host = api_server_config.host

    prometheus_multiproc_dir = providers.Factory[str](
        os.path.join,
        bentoml_container.bentoml_home,
        "prometheus_multiproc_dir",
    )

    @providers.SingletonFactory
    @staticmethod
    def metrics_client(
        multiproc_dir: str = Provide[prometheus_multiproc_dir],
        namespace: str = Provide[api_server_config.metrics.namespace],
    ) -> "PrometheusClient":
        from ..server.metrics.prometheus import PrometheusClient

        return PrometheusClient(
            multiproc_dir=multiproc_dir,
            namespace=namespace,
        )

    @providers.SingletonFactory
    @staticmethod
    def tracer_provider(
        tracer_type: str = Provide[config.tracing.type],
        sample_rate: t.Optional[float] = Provide[config.tracing.sample_rate],
        zipkin_server_url: t.Optional[str] = Provide[config.tracing.zipkin.url],
        jaeger_server_address: t.Optional[str] = Provide[config.tracing.jaeger.address],
        jaeger_server_port: t.Optional[int] = Provide[config.tracing.jaeger.port],
    ):
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.trace.export import BatchSpanProcessor

        from ..utils.telemetry import ParentBasedTraceIdRatio

        if sample_rate is None:
            sample_rate = 0.0

        provider = TracerProvider(
            sampler=ParentBasedTraceIdRatio(sample_rate),
            # resource: Resource = Resource.create({}),
            # shutdown_on_exit: bool = True,
            # active_span_processor: Union[
            # SynchronousMultiSpanProcessor, ConcurrentMultiSpanProcessor
            # ] = None,
            # id_generator: IdGenerator = None,
        )

        if tracer_type == "zipkin" and zipkin_server_url is not None:
            from opentelemetry.exporter.zipkin.json import ZipkinExporter

            exporter = ZipkinExporter(
                endpoint=zipkin_server_url,
            )
            provider.add_span_processor(BatchSpanProcessor(exporter))
            return provider
        elif (
            tracer_type == "jaeger"
            and jaeger_server_address is not None
            and jaeger_server_port is not None
        ):
            from opentelemetry.exporter.jaeger.thrift import JaegerExporter

            exporter = JaegerExporter(
                agent_host_name=jaeger_server_address,
                agent_port=jaeger_server_port,
            )
            provider.add_span_processor(BatchSpanProcessor(exporter))
            return provider
        else:
            return provider

    # Mapping from runner name to RunnerApp file descriptor
    remote_runner_mapping = providers.Static[t.Dict[str, str]]({})
    plasma_db = providers.Static[t.Optional["ext.PlasmaClient"]](None)


DeploymentContainer = DeploymentContainerClass()
