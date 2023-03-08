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
import schema as s
from simple_di import Provide
from simple_di import providers
from deepmerge.merger import Merger

from . import expand_env_var
from ..utils import split_with_quotes
from ..utils import validate_or_create_dir
from .helpers import flatten_dict
from .helpers import load_config_file
from .helpers import get_default_config
from .helpers import import_configuration_spec
from ..context import component_context
from ..resource import CpuResource
from ..resource import system_resources
from ...exceptions import BentoMLConfigException
from ..utils.unflatten import unflatten

if TYPE_CHECKING:
    from fs.base import FS

    from .. import external_typing as ext
    from ..models import ModelStore
    from ..utils.analytics import ServeInfo
    from ..server.metrics.prometheus import PrometheusClient

    SerializationStrategy = t.Literal["EXPORT_BENTO", "LOCAL_BENTO", "REMOTE_BENTO"]

config_merger = Merger(
    # merge dicts
    type_strategies=[(dict, "merge")],
    # override all other types
    fallback_strategies=["override"],
    # override conflicting types
    type_conflict_strategies=["override"],
)

logger = logging.getLogger(__name__)


class BentoMLConfiguration:
    def __init__(
        self,
        override_config_file: str | None = None,
        override_config_values: str | None = None,
        *,
        validate_schema: bool = True,
        use_version: int = 1,
    ):
        # Load default configuration with latest version.
        self.config = get_default_config(version=use_version)
        spec_module = import_configuration_spec(version=use_version)

        # User override configuration
        if override_config_file is not None:
            logger.info(
                "Applying user config override from path: %s" % override_config_file
            )
            override = load_config_file(override_config_file)
            if "version" not in override:
                # If users does not define a version, we then by default assume they are using v1
                # and we will migrate it to latest version
                logger.debug(
                    "User config does not define a version, assuming given config is version %d..."
                    % use_version
                )
                current = use_version
            else:
                current = override["version"]
            migration = getattr(import_configuration_spec(current), "migration", None)
            # Running migration layer if it exists
            if migration is not None:
                override = migration(override_config=dict(flatten_dict(override)))
            config_merger.merge(self.config, override)

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
                    split_with_quotes(line, sep="=", quote='"')
                    for line in lines
                    if line.strip()
                ]
            }
            # Note that this values will only support latest version of configuration,
            # as there is no way for us to infer what values user can pass in.
            # however, if users pass in a version inside this value, we will that to migrate up
            # if possible
            override_version = override_config_map.get("version", use_version)
            logger.debug(
                "Found defined 'version=%d' in BENTOML_CONFIG_OPTIONS."
                % override_version
            )
            migration = getattr(
                import_configuration_spec(override_version), "migration", None
            )
            # Running migration layer if it exists
            if migration is not None:
                override_config_map = migration(override_config=override_config_map)
            # Previous behaviour, before configuration versioning.
            try:
                override = unflatten(override_config_map)
            except ValueError as e:
                raise BentoMLConfigException(
                    f"Failed to parse config options from the env var:\n{e}.\n*** Note: You can use '\"' to quote the key if it contains special characters. ***"
                ) from None
            config_merger.merge(self.config, override)

        if override_config_file is not None or override_config_values is not None:
            self._finalize()

        if validate_schema:
            try:
                spec_module.SCHEMA.validate(self.config)
            except s.SchemaError as e:
                raise BentoMLConfigException(
                    f"Invalid configuration file was given:\n{e}"
                ) from None

    def _finalize(self):
        RUNNER_CFG_KEYS = ["batching", "resources", "logging", "metrics", "timeout"]
        global_runner_cfg = {k: self.config["runners"][k] for k in RUNNER_CFG_KEYS}
        custom_runners_cfg = dict(
            filter(
                lambda kv: kv[0] not in RUNNER_CFG_KEYS,
                self.config["runners"].items(),
            )
        )
        if custom_runners_cfg:
            for runner_name, runner_cfg in custom_runners_cfg.items():
                # key is a runner name
                if runner_cfg.get("resources") == "system":
                    runner_cfg["resources"] = system_resources()
                self.config["runners"][runner_name] = config_merger.merge(
                    deepcopy(global_runner_cfg),
                    runner_cfg,
                )

    def to_dict(self) -> providers.ConfigDictType:
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
        envs = os.path.join(home, "envs")
        tmp_bentos = os.path.join(home, "tmp")

        validate_or_create_dir(home, bentos, models, envs, tmp_bentos)
        return home

    @providers.SingletonFactory
    @staticmethod
    def tmp_bento_store_dir(bentoml_home: str = Provide[bentoml_home]):
        return os.path.join(bentoml_home, "tmp")

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
    def env_store_dir(bentoml_home: str = Provide[bentoml_home]):
        return os.path.join(bentoml_home, "envs")

    @providers.SingletonFactory
    @staticmethod
    def env_store(bentoml_home: str = Provide[bentoml_home]) -> FS:
        import fs

        return fs.open_fs(os.path.join(bentoml_home, "envs"))

    @providers.SingletonFactory
    @staticmethod
    def bento_store(base_dir: str = Provide[bento_store_dir]):
        from ..bento import BentoStore

        return BentoStore(base_dir)

    @providers.SingletonFactory
    @staticmethod
    def tmp_bento_store(base_dir: str = Provide[tmp_bento_store_dir]):
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
    ssl = api_server_config.ssl

    development_mode = providers.Static(True)
    serialization_strategy: SerializationStrategy = providers.Static("EXPORT_BENTO")

    @providers.SingletonFactory
    @staticmethod
    def yatai_client():
        from ..yatai_client import YataiClient

        return YataiClient()

    @providers.SingletonFactory
    @staticmethod
    def serve_info() -> ServeInfo:
        from ..utils.analytics import get_serve_info

        return get_serve_info()

    cors = http.cors

    @providers.SingletonFactory
    @staticmethod
    def access_control_options(
        allow_origins: list[str]
        | str
        | None = Provide[cors.access_control_allow_origins],
        allow_origin_regex: str
        | None = Provide[cors.access_control_allow_origin_regex],
        allow_credentials: bool | None = Provide[cors.access_control_allow_credentials],
        allow_methods: list[str]
        | str
        | None = Provide[cors.access_control_allow_methods],
        allow_headers: list[str]
        | str
        | None = Provide[cors.access_control_allow_headers],
        max_age: int | None = Provide[cors.access_control_max_age],
        expose_headers: list[str]
        | str
        | None = Provide[cors.access_control_expose_headers],
    ) -> dict[str, list[str] | str | int]:

        if isinstance(allow_origins, str):
            allow_origins = [allow_origins]

        if isinstance(allow_headers, str):
            allow_headers = [allow_headers]

        return {
            k: v
            for k, v in {
                "allow_origins": allow_origins,
                "allow_origin_regex": allow_origin_regex,
                "allow_credentials": allow_credentials,
                "allow_methods": allow_methods,
                "allow_headers": allow_headers,
                "max_age": max_age,
                "expose_headers": expose_headers,
            }.items()
            if v is not None
        }

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

    tracing = config.tracing

    @providers.SingletonFactory
    @staticmethod
    def tracer_provider(
        tracer_type: str = Provide[tracing.exporter_type],
        sample_rate: float | None = Provide[tracing.sample_rate],
        timeout: int | None = Provide[tracing.timeout],
        max_tag_value_length: int | None = Provide[tracing.max_tag_value_length],
        zipkin: dict[str, t.Any] = Provide[tracing.zipkin],
        jaeger: dict[str, t.Any] = Provide[tracing.jaeger],
        otlp: dict[str, t.Any] = Provide[tracing.otlp],
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
        if sample_rate == 0.0:
            logger.debug(
                "'tracing.sample_rate' is set to zero. No traces will be collected. Please refer to https://docs.bentoml.org/en/latest/guides/tracing.html for more details."
            )
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
        # create tracer provider
        provider = TracerProvider(
            sampler=ParentBasedTraceIdRatio(sample_rate),
            resource=Resource.create(resource),
        )
        if tracer_type == "zipkin" and any(zipkin.values()):
            from opentelemetry.exporter.zipkin.json import ZipkinExporter

            exporter = ZipkinExporter(
                endpoint=zipkin.get("endpoint"),
                local_node_ipv4=zipkin.get("local_node_ipv4"),
                local_node_ipv6=zipkin.get("local_node_ipv6"),
                local_node_port=zipkin.get("local_node_port"),
                max_tag_value_length=max_tag_value_length,
                timeout=timeout,
            )
        elif tracer_type == "jaeger" and any(jaeger.values()):
            protocol = jaeger.get("protocol")
            if protocol == "thrift":
                from opentelemetry.exporter.jaeger.thrift import JaegerExporter
            elif protocol == "grpc":
                from opentelemetry.exporter.jaeger.proto.grpc import JaegerExporter
            else:
                raise InvalidArgument(
                    f"Invalid 'api_server.tracing.jaeger.protocol' value: {protocol}"
                ) from None
            exporter = JaegerExporter(
                collector_endpoint=jaeger.get("collector_endpoint"),
                max_tag_value_length=max_tag_value_length,
                timeout=timeout,
                **jaeger[protocol],
            )
        elif tracer_type == "otlp" and any(otlp.values()):
            protocol = otlp.get("protocol")
            if protocol == "grpc":
                from opentelemetry.exporter.otlp.proto.grpc import trace_exporter
            elif protocol == "http":
                from opentelemetry.exporter.otlp.proto.http import trace_exporter
            else:
                raise InvalidArgument(
                    f"Invalid 'api_server.tracing.jaeger.protocol' value: {protocol}"
                ) from None
            exporter = trace_exporter.OTLPSpanExporter(
                endpoint=otlp.get("endpoint", None),
                compression=otlp.get("compression", None),
                timeout=timeout,
                **otlp[protocol],
            )
        elif tracer_type == "in_memory":
            # This will be used during testing, user shouldn't use this otherwise.
            # We won't document this in documentation.
            from opentelemetry.sdk.trace.export import in_memory_span_exporter

            exporter = in_memory_span_exporter.InMemorySpanExporter()
        else:
            return provider
        # When exporter is set
        provider.add_span_processor(BatchSpanProcessor(exporter))
        return provider

    @providers.SingletonFactory
    @staticmethod
    def tracing_excluded_urls(
        excluded_urls: str | list[str] | None = Provide[tracing.excluded_urls],
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
        duration: dict[str, t.Any] = Provide[api_server_config.metrics.duration]
    ) -> tuple[float, ...]:
        """
        Returns a tuple of duration buckets in seconds. If not explicitly configured,
        the Prometheus default is returned; otherwise, a set of exponential buckets
        generated based on the configuration is returned.
        """
        from ..utils.metrics import INF
        from ..utils.metrics import exponential_buckets

        if "buckets" in duration:
            return tuple(duration["buckets"]) + (INF,)
        else:
            if len(set(duration) - {"min", "max", "factor"}) == 0:
                return exponential_buckets(
                    duration["min"], duration["factor"], duration["max"]
                )
            raise BentoMLConfigException(
                f"Keys 'min', 'max', and 'factor' are required for 'duration' configuration, '{duration!r}'."
            ) from None

    @providers.SingletonFactory
    @staticmethod
    def logging_formatting(
        cfg: dict[str, t.Any] = Provide[api_server_config.logging.access.format],
    ) -> dict[str, str]:
        return cfg


BentoMLContainer = _BentoMLContainerClass()
