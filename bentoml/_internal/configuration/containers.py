from __future__ import annotations

import os
import math
import uuid
import typing as t
import logging
from copy import deepcopy
from typing import TYPE_CHECKING
from dataclasses import dataclass

import yaml
from schema import Or
from schema import And
from schema import Use
from schema import Schema
from schema import Optional
from schema import SchemaError
from simple_di import Provide
from simple_di import providers
from deepmerge.merger import Merger

from . import expand_env_var
from ..utils import split_with_quotes
from ..utils import validate_or_create_dir
from ..context import component_context
from ..resource import CpuResource
from ..resource import system_resources
from ...exceptions import BentoMLConfigException
from ..utils.unflatten import unflatten

if TYPE_CHECKING:
    from bentoml._internal.models import ModelStore

    from .. import external_typing as ext
    from ..utils.analytics import ServeInfo
    from ..server.metrics.prometheus import PrometheusClient


config_merger = Merger(
    # merge dicts
    [(dict, "merge")],
    # override all other types
    ["override"],
    # override conflicting types
    ["override"],
)

logger = logging.getLogger(__name__)

_check_tracing_type: t.Callable[[str], bool] = lambda s: s in (
    "zipkin",
    "jaeger",
    "otlp",
)
_check_otlp_protocol: t.Callable[[str], bool] = lambda s: s in (
    "grpc",
    "http",
)
_larger_than: t.Callable[[int | float], t.Callable[[int | float], bool]] = (
    lambda target: lambda val: val > target
)
_larger_than_zero: t.Callable[[int | float], bool] = _larger_than(0)


def _check_sample_rate(sample_rate: float) -> None:
    if sample_rate == 0.0:
        logger.warning(
            "Tracing enabled, but sample_rate is unset or zero. No traces will be collected. Please refer to https://docs.bentoml.org/en/latest/guides/tracing.html for more details."
        )


def _is_ip_address(addr: str) -> bool:
    import socket

    try:
        socket.inet_aton(addr)
        return True
    except socket.error:
        return False


RUNNER_CFG_KEYS = ["batching", "resources", "logging", "metrics", "timeout"]

RUNNER_CFG_SCHEMA = {
    Optional("batching"): {
        Optional("enabled"): bool,
        Optional("max_batch_size"): And(int, _larger_than_zero),
        Optional("max_latency_ms"): And(int, _larger_than_zero),
    },
    # note there is a distinction between being unset and None here; if set to 'None'
    # in configuration for a specific runner, it will override the global configuration
    Optional("resources"): Or({Optional(str): object}, lambda s: s == "system", None),  # type: ignore (incomplete schema typing)
    Optional("logging"): {
        # TODO add logging level configuration
        Optional("access"): {
            Optional("enabled"): bool,
            Optional("request_content_length"): Or(bool, None),
            Optional("request_content_type"): Or(bool, None),
            Optional("response_content_length"): Or(bool, None),
            Optional("response_content_type"): Or(bool, None),
        },
    },
    Optional("metrics"): {
        "enabled": bool,
        "namespace": str,
    },
    Optional("timeout"): And(int, _larger_than_zero),
}

SCHEMA = Schema(
    {
        "api_server": {
            "workers": Or(And(int, _larger_than_zero), None),
            "timeout": And(int, _larger_than_zero),
            "backlog": And(int, _larger_than(64)),
            Optional("ssl"): {
                Optional("certfile"): Or(str, None),
                Optional("keyfile"): Or(str, None),
                Optional("keyfile_password"): Or(str, None),
                Optional("version"): Or(And(int, _larger_than_zero), None),
                Optional("cert_reqs"): Or(int, None),
                Optional("ca_certs"): Or(str, None),
                Optional("ciphers"): Or(str, None),
            },
            "metrics": {
                "enabled": bool,
                "namespace": str,
                Optional("duration"): {
                    Optional("min"): And(float, _larger_than_zero),
                    Optional("max"): And(float, _larger_than_zero),
                    Optional("factor"): And(float, _larger_than(1.0)),
                },
            },
            "logging": {
                # TODO add logging level configuration
                "access": {
                    "enabled": bool,
                    "request_content_length": Or(bool, None),
                    "request_content_type": Or(bool, None),
                    "response_content_length": Or(bool, None),
                    "response_content_type": Or(bool, None),
                    "format": {
                        "trace_id": str,
                        "span_id": str,
                    },
                },
            },
            "http": {
                "host": And(str, _is_ip_address),
                "port": And(int, _larger_than_zero),
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
            "grpc": {
                "host": And(str, _is_ip_address),
                "port": And(int, _larger_than_zero),
                "metrics": {
                    "port": And(int, _larger_than_zero),
                    "host": And(str, _is_ip_address),
                },
                "reflection": {"enabled": bool},
                "max_concurrent_streams": Or(int, None),
                "max_message_length": Or(int, None),
                "maximum_concurrent_rpcs": Or(int, None),
            },
        },
        "runners": {
            **RUNNER_CFG_SCHEMA,
            Optional(str): RUNNER_CFG_SCHEMA,  # type: ignore (incomplete schema typing)
        },
        "tracing": {
            "type": Or(And(str, Use(str.lower), _check_tracing_type), None),
            "sample_rate": Or(And(float, lambda i: i >= 0 and i <= 1), None),
            "excluded_urls": Or([str], str, None),
            Optional("zipkin"): {"url": Or(str, None)},
            Optional("jaeger"): {"address": Or(str, None), "port": Or(int, None)},
            Optional("otlp"): {
                "protocol": Or(And(str, Use(str.lower), _check_otlp_protocol), None),
                "url": Or(str, None),
            },
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

_WARNING_MESSAGE = (
    "field 'api_server.%s' is deprecated and has been renamed to 'api_server.http.%s'"
)


class BentoMLConfiguration:
    def __init__(
        self,
        override_config_file: t.Optional[str] = None,
        override_config_values: t.Optional[str] = None,
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
            logger.info("Applying user config override from %s" % override_config_file)
            if not os.path.exists(override_config_file):
                raise BentoMLConfigException(
                    f"Config file {override_config_file} not found"
                )
            with open(override_config_file, "rb") as f:
                override_config: dict[str, t.Any] = yaml.safe_load(f)

            # compatibility layer with old configuration pre gRPC features
            # api_server.[cors|port|host] -> api_server.http.$^
            if "api_server" in override_config:
                user_api_config = override_config["api_server"]
                # max_request_size is deprecated
                if "max_request_size" in user_api_config:
                    logger.warning(
                        "'api_server.max_request_size' is deprecated and has become obsolete."
                    )
                    user_api_config.pop("max_request_size")
                # check if user are using older configuration
                if "http" not in user_api_config:
                    user_api_config["http"] = {}
                # then migrate these fields to newer configuration fields.
                for field in ["port", "host", "cors"]:
                    if field in user_api_config:
                        old_field = user_api_config.pop(field)
                        user_api_config["http"][field] = old_field
                        logger.warning(_WARNING_MESSAGE, field, field)

                config_merger.merge(override_config["api_server"], user_api_config)

                assert all(
                    key not in override_config["api_server"]
                    for key in ["cors", "max_request_size", "host", "port"]
                )

            config_merger.merge(self.config, override_config)

        if override_config_values is not None:
            logger.info(
                "Applying user config override from ENV VAR: %s", override_config_values
            )
            lines = split_with_quotes(
                override_config_values,
                sep=r"\s+",
                quote='"',
                use_regex=True,
            )
            override_config_map = {
                k: yaml.safe_load(v)
                for k, v in [
                    split_with_quotes(line, sep="=", quote='"') for line in lines
                ]
            }
            try:
                override_config = unflatten(override_config_map)
            except ValueError as e:
                raise BentoMLConfigException(
                    f"Failed to parse config options from the env var: {e}. \n *** Note: You can use '\"' to quote the key if it contains special characters. ***"
                ) from None
            config_merger.merge(self.config, override_config)

        if override_config_file is not None or override_config_values is not None:
            self._finalize()

        if validate_schema:
            try:
                SCHEMA.validate(self.config)
            except SchemaError as e:
                raise BentoMLConfigException(
                    "Invalid configuration file was given."
                ) from e

    def _finalize(self):
        global_runner_cfg = {k: self.config["runners"][k] for k in RUNNER_CFG_KEYS}
        for key in self.config["runners"]:
            if key not in RUNNER_CFG_KEYS:
                runner_cfg = self.config["runners"][key]
                # key is a runner name
                if runner_cfg.get("resources") == "system":
                    runner_cfg["resources"] = system_resources()
                self.config["runners"][key] = config_merger.merge(
                    deepcopy(global_runner_cfg),
                    runner_cfg,
                )

    def override(self, keys: t.List[str], value: t.Any):
        if keys is None:
            raise BentoMLConfigException(
                "Configuration override key is None."
            ) from None
        if len(keys) == 0:
            raise BentoMLConfigException(
                "Configuration override key is empty."
            ) from None
        if value is None:
            return

        c = self.config
        for key in keys[:-1]:
            if key not in c:
                raise BentoMLConfigException(
                    "Configuration override key is invalid, %s" % keys
                ) from None
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
class _BentoMLContainerClass:

    config = providers.Configuration()

    @providers.SingletonFactory
    @staticmethod
    def bentoml_home() -> str:
        home = expand_env_var(
            str(
                os.environ.get(
                    "BENTOML_HOME", os.path.join(os.path.expanduser("~"), "bentoml")
                )
            )
        )
        bentos = os.path.join(home, "bentos")
        models = os.path.join(home, "models")

        validate_or_create_dir(home, bentos, models)
        return home

    @providers.SingletonFactory
    @staticmethod
    def bento_store_dir(bentoml_home: str = Provide[bentoml_home]):
        return os.path.join(bentoml_home, "bentos")

    @providers.SingletonFactory
    @staticmethod
    def model_store_dir(bentoml_home: str = Provide[bentoml_home]):
        return os.path.join(bentoml_home, "models")

    @providers.SingletonFactory
    @staticmethod
    def bento_store(base_dir: str = Provide[bento_store_dir]):
        from ..bento import BentoStore

        return BentoStore(base_dir)

    @providers.SingletonFactory
    @staticmethod
    def model_store(base_dir: str = Provide[model_store_dir]) -> "ModelStore":
        from ..models import ModelStore

        return ModelStore(base_dir)

    @providers.SingletonFactory
    @staticmethod
    def session_id() -> str:
        return uuid.uuid1().hex

    api_server_config = config.api_server
    runners_config = config.runners

    grpc = api_server_config.grpc
    http = api_server_config.http

    development_mode = providers.Static(True)

    @providers.SingletonFactory
    @staticmethod
    def serve_info() -> ServeInfo:
        from ..utils.analytics import get_serve_info

        return get_serve_info()

    @providers.SingletonFactory
    @staticmethod
    def access_control_options(
        allow_origins: str | None = Provide[http.cors.access_control_allow_origin],
        allow_credentials: bool
        | None = Provide[http.cors.access_control_allow_credentials],
        expose_headers: list[str]
        | str
        | None = Provide[http.cors.access_control_expose_headers],
        allow_methods: list[str]
        | str
        | None = Provide[http.cors.access_control_allow_methods],
        allow_headers: list[str]
        | str
        | None = Provide[http.cors.access_control_allow_headers],
        max_age: int | None = Provide[http.cors.access_control_max_age],
    ) -> dict[str, list[str] | str | int]:
        kwargs = dict(
            allow_origins=allow_origins,
            allow_credentials=allow_credentials,
            expose_headers=expose_headers,
            allow_methods=allow_methods,
            allow_headers=allow_headers,
            max_age=max_age,
        )

        filtered_kwargs: dict[str, list[str] | str | int] = {
            k: v for k, v in kwargs.items() if v is not None
        }
        return filtered_kwargs

    api_server_workers = providers.Factory[int](
        lambda workers: workers or math.ceil(CpuResource.from_system()),
        api_server_config.workers,
    )

    prometheus_multiproc_dir = providers.Factory[str](
        os.path.join,
        bentoml_home,
        "prometheus_multiproc_dir",
    )

    @providers.SingletonFactory
    @staticmethod
    def metrics_client(
        multiproc_dir: str = Provide[prometheus_multiproc_dir],
    ) -> PrometheusClient:
        from ..server.metrics.prometheus import PrometheusClient

        return PrometheusClient(multiproc_dir=multiproc_dir)

    @providers.SingletonFactory
    @staticmethod
    def tracer_provider(
        tracer_type: str = Provide[config.tracing.type],
        sample_rate: t.Optional[float] = Provide[config.tracing.sample_rate],
        zipkin_server_url: t.Optional[str] = Provide[config.tracing.zipkin.url],
        jaeger_server_address: t.Optional[str] = Provide[config.tracing.jaeger.address],
        jaeger_server_port: t.Optional[int] = Provide[config.tracing.jaeger.port],
        otlp_server_protocol: t.Optional[str] = Provide[config.tracing.otlp.protocol],
        otlp_server_url: t.Optional[str] = Provide[config.tracing.otlp.url],
    ):
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.resources import Resource
        from opentelemetry.sdk.resources import SERVICE_NAME
        from opentelemetry.sdk.resources import SERVICE_VERSION
        from opentelemetry.sdk.resources import SERVICE_NAMESPACE
        from opentelemetry.sdk.resources import SERVICE_INSTANCE_ID
        from opentelemetry.sdk.trace.export import BatchSpanProcessor
        from opentelemetry.sdk.environment_variables import OTEL_SERVICE_NAME
        from opentelemetry.sdk.environment_variables import OTEL_RESOURCE_ATTRIBUTES

        from ...exceptions import InvalidArgument
        from ..utils.telemetry import ParentBasedTraceIdRatio

        if sample_rate is None:
            sample_rate = 0.0

        resource = {}

        # User can optionally configure the resource with the following environment variables. Only
        # configure resource if user has not explicitly configured it.
        if (
            OTEL_SERVICE_NAME not in os.environ
            and OTEL_RESOURCE_ATTRIBUTES not in os.environ
        ):
            if component_context.component_name:
                resource[SERVICE_NAME] = component_context.component_name
            if component_context.component_index:
                resource[SERVICE_INSTANCE_ID] = component_context.component_index
            if component_context.bento_name:
                resource[SERVICE_NAMESPACE] = component_context.bento_name
            if component_context.bento_version:
                resource[SERVICE_VERSION] = component_context.bento_version

        provider = TracerProvider(
            sampler=ParentBasedTraceIdRatio(sample_rate),
            resource=Resource.create(resource),
        )

        if tracer_type == "zipkin" and zipkin_server_url is not None:
            from opentelemetry.exporter.zipkin.json import ZipkinExporter

            exporter = ZipkinExporter(endpoint=zipkin_server_url)
            provider.add_span_processor(BatchSpanProcessor(exporter))
            _check_sample_rate(sample_rate)
            return provider
        elif (
            tracer_type == "jaeger"
            and jaeger_server_address is not None
            and jaeger_server_port is not None
        ):
            from opentelemetry.exporter.jaeger.thrift import JaegerExporter

            exporter = JaegerExporter(
                agent_host_name=jaeger_server_address, agent_port=jaeger_server_port
            )
            provider.add_span_processor(BatchSpanProcessor(exporter))
            _check_sample_rate(sample_rate)
            return provider
        elif (
            tracer_type == "otlp"
            and otlp_server_protocol is not None
            and otlp_server_url is not None
        ):
            if otlp_server_protocol == "grpc":
                from opentelemetry.exporter.otlp.proto.grpc import trace_exporter

            elif otlp_server_protocol == "http":
                from opentelemetry.exporter.otlp.proto.http import trace_exporter
            else:
                raise InvalidArgument(
                    f"Invalid otlp protocol: {otlp_server_protocol}"
                ) from None
            exporter = trace_exporter.OTLPSpanExporter(endpoint=otlp_server_url)
            provider.add_span_processor(BatchSpanProcessor(exporter))
            _check_sample_rate(sample_rate)
            return provider
        else:
            return provider

    @providers.SingletonFactory
    @staticmethod
    def tracing_excluded_urls(
        excluded_urls: str | list[str] | None = Provide[config.tracing.excluded_urls],
    ):
        from opentelemetry.util.http import ExcludeList
        from opentelemetry.util.http import parse_excluded_urls

        if isinstance(excluded_urls, list):
            return ExcludeList(excluded_urls)
        elif isinstance(excluded_urls, str):
            return parse_excluded_urls(excluded_urls)
        else:
            return ExcludeList([])

    # Mapping from runner name to RunnerApp file descriptor
    remote_runner_mapping = providers.Static[t.Dict[str, str]]({})
    plasma_db = providers.Static[t.Optional["ext.PlasmaClient"]](None)

    @providers.SingletonFactory
    @staticmethod
    def duration_buckets(
        metrics: dict[str, t.Any] = Provide[api_server_config.metrics]
    ) -> tuple[float, ...]:
        """
        Returns a tuple of duration buckets in seconds. If not explicitly configured,
        the Prometheus default is returned; otherwise, a set of exponential buckets
        generated based on the configuration is returned.
        """
        from ..utils.metrics import DEFAULT_BUCKET
        from ..utils.metrics import exponential_buckets

        if "duration" in metrics:
            duration: dict[str, float] = metrics["duration"]
            if duration.keys() >= {"min", "max", "factor"}:
                return exponential_buckets(
                    duration["min"], duration["factor"], duration["max"]
                )
            raise BentoMLConfigException(
                "Keys 'min', 'max', and 'factor' are required for "
                f"'duration' configuration, '{duration}'."
            )
        return DEFAULT_BUCKET

    @providers.SingletonFactory
    @staticmethod
    def logging_formatting(
        cfg: dict[str, t.Any] = Provide[api_server_config.logging.access.format],
    ) -> dict[str, str]:
        return cfg


BentoMLContainer = _BentoMLContainerClass()
