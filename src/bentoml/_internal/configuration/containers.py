from __future__ import annotations

import logging
import math
import os
import typing as t
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import schema as s
import yaml
from deepmerge.merger import Merger
from simple_di import Provide
from simple_di import providers

from ...exceptions import BentoMLConfigException
from ..context import server_context
from ..context import trace_context
from ..resource import CpuResource
from ..utils import split_with_quotes
from ..utils.filesystem import validate_or_create_dir
from ..utils.unflatten import unflatten
from . import expand_env_var
from . import load_config
from .helpers import expand_env_var_in_values
from .helpers import flatten_dict
from .helpers import get_default_config
from .helpers import import_configuration_spec
from .helpers import load_config_file

if TYPE_CHECKING:
    from fs.base import FS

    from .. import external_typing as ext
    from ..bento import BentoStore
    from ..cloud import BentoCloudClient
    from ..cloud.client import RestApiClient
    from ..models import ModelStore
    from ..server.metrics.prometheus import PrometheusClient
    from ..utils.analytics import ServeInfo

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
        override_defaults: dict[str, t.Any] | None = None,
        override_config_json: dict[str, t.Any] | None = None,
        *,
        validate_schema: bool = True,
        use_version: int = 1,
    ):
        # Load default configuration with latest version.
        self.config = get_default_config(version=use_version)
        spec_module = import_configuration_spec(version=use_version)
        migration = getattr(spec_module, "migration", None)

        if override_defaults:
            if migration is not None:
                override_defaults = migration(
                    override_config=dict(flatten_dict(override_defaults)),
                )
            config_merger.merge(self.config, override_defaults)

        # User override configuration
        if override_config_file is not None:
            logger.info(
                "Applying user config override from path: %s" % override_config_file
            )
            override = load_config_file(override_config_file)
            # Running migration layer if it exists
            if migration is not None:
                override = migration(
                    override_config=dict(flatten_dict(override)),
                )
            config_merger.merge(self.config, override)

        if override_config_json is not None:
            logger.info(
                "Applying user config override from json: %s" % override_config_json
            )
            # Running migration layer if it exists
            if migration is not None:
                override_config_json = migration(
                    override_config=dict(flatten_dict(override_config_json)),
                )
            config_merger.merge(self.config, override_config_json)

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

        if finalize_config := getattr(spec_module, "finalize_config", None):
            finalize_config(self.config)
        expand_env_var_in_values(self.config)

        if validate_schema:
            try:
                spec_module.SCHEMA.validate(self.config)
            except s.SchemaError as e:
                raise BentoMLConfigException(
                    f"Invalid configuration file was given:\n{e}"
                ) from None

    def to_dict(self) -> providers.ConfigDictType:
        return t.cast(providers.ConfigDictType, self.config)


@dataclass
class _BentoMLContainerClass:
    config = providers.Configuration()
    model_aliases = providers.Static({})

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
    def result_store_file(bentoml_home: str = Provide[bentoml_home]) -> str:
        path = os.getenv(
            "BENTOML_RESULT_STORE", os.path.join(bentoml_home, "task_result.db")
        )
        return (
            os.path.realpath(os.path.expanduser(path)) if path != ":memory:" else path
        )

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

        from ..utils.uri import encode_path_for_uri

        return fs.open_fs(encode_path_for_uri(os.path.join(bentoml_home, "envs")))

    @providers.SingletonFactory
    @staticmethod
    def bento_store(base_dir: str = Provide[bento_store_dir]) -> BentoStore:
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

    @providers.SingletonFactory
    @staticmethod
    def cloud_config(bentoml_home: str = Provide[bentoml_home]) -> Path:
        return Path(bentoml_home) / ".yatai.yaml"

    api_server_config = config.api_server
    runners_config = config.runners

    grpc = api_server_config.grpc
    http = api_server_config.http
    ssl = api_server_config.ssl

    development_mode = providers.Static(True)
    serialization_strategy: providers.Static[SerializationStrategy] = providers.Static(
        "EXPORT_BENTO"
    )
    worker_index: providers.Static[int] = providers.Static(0)

    @providers.SingletonFactory
    @staticmethod
    def serve_info() -> ServeInfo:
        from ..utils.analytics import get_serve_info

        return get_serve_info()

    cors = http.cors

    @providers.SingletonFactory
    @staticmethod
    def access_control_options(
        allow_origins: list[str] | str | None = Provide[
            cors.access_control_allow_origins
        ],
        allow_origin_regex: str | None = Provide[
            cors.access_control_allow_origin_regex
        ],
        allow_credentials: bool | None = Provide[cors.access_control_allow_credentials],
        allow_methods: list[str] | str | None = Provide[
            cors.access_control_allow_methods
        ],
        allow_headers: list[str] | str | None = Provide[
            cors.access_control_allow_headers
        ],
        max_age: int | None = Provide[cors.access_control_max_age],
        expose_headers: list[str] | str | None = Provide[
            cors.access_control_expose_headers
        ],
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
    def metrics_client() -> PrometheusClient:
        from ..server.metrics.prometheus import PrometheusClient

        return PrometheusClient()

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
        from opentelemetry.sdk.resources import SERVICE_INSTANCE_ID
        from opentelemetry.sdk.resources import SERVICE_NAME
        from opentelemetry.sdk.resources import SERVICE_NAMESPACE
        from opentelemetry.sdk.resources import SERVICE_VERSION
        from opentelemetry.sdk.resources import OTELResourceDetector
        from opentelemetry.sdk.resources import Resource
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.trace.export import BatchSpanProcessor

        from ...exceptions import InvalidArgument
        from ..utils.telemetry import ParentBasedTraceIdRatio

        if sample_rate is None:
            sample_rate = 0.0
        if sample_rate == 0.0:
            logger.debug(
                "'tracing.sample_rate' is set to zero. No traces will be collected. Please refer to https://docs.bentoml.com/en/latest/guides/tracing.html for more details."
            )

        # User can optionally configure the resource with the following environment variables. Only
        # configure resource if user has not explicitly configured it.
        system_otel_resources: Resource = OTELResourceDetector().detect()

        _resource = {}
        if server_context.service_name:
            _resource[SERVICE_NAME] = server_context.service_name
        if server_context.worker_index:
            _resource[SERVICE_INSTANCE_ID] = server_context.worker_index
        if server_context.bento_name:
            _resource[SERVICE_NAMESPACE] = server_context.bento_name
        if server_context.bento_version:
            _resource[SERVICE_VERSION] = server_context.bento_version

        bentoml_resource = Resource.create(_resource)

        resources = bentoml_resource.merge(system_otel_resources)
        trace_context.service_name = resources.attributes.get(SERVICE_NAME)

        # create tracer provider
        provider = TracerProvider(
            sampler=ParentBasedTraceIdRatio(sample_rate), resource=resources
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
        duration: dict[str, t.Any] = Provide[api_server_config.metrics.duration],
    ) -> tuple[float, ...]:
        """
        Returns a tuple of duration buckets in seconds. If not explicitly configured,
        the Prometheus default is returned; otherwise, a set of exponential buckets
        generated based on the configuration is returned.
        """
        from ..utils.metrics import INF
        from ..utils.metrics import exponential_buckets

        if None not in (
            duration.get("min"),
            duration.get("max"),
            duration.get("factor"),
        ):
            return exponential_buckets(
                duration["min"], duration["factor"], duration["max"]
            )
        elif "buckets" in duration:
            return tuple(duration["buckets"]) + (INF,)
        else:
            raise BentoMLConfigException(
                "Either `buckets` or `min`, `max`, and `factor` must be set in `api_server.metrics.duration`"
            )

    @providers.SingletonFactory
    @staticmethod
    def logging_formatting(
        cfg: dict[str, t.Any] = Provide[api_server_config.logging.access.format],
    ) -> dict[str, str]:
        return cfg

    @providers.SingletonFactory
    @staticmethod
    def enabled_features() -> list[str]:
        return os.getenv("BENTOML_ENABLE_FEATURES", "").split(",")

    @property
    def new_index(self) -> bool:
        return "new_index" in self.enabled_features.get()

    @providers.SingletonFactory
    @staticmethod
    def cloud_context() -> str | None:
        return os.getenv("BENTOML_CLOUD_CONTEXT")

    @providers.SingletonFactory
    @staticmethod
    def rest_api_client(
        context: str | None = Provide[cloud_context],
    ) -> RestApiClient:
        from ..cloud.config import get_rest_api_client

        return get_rest_api_client(context)

    @providers.SingletonFactory
    @staticmethod
    def bentocloud_client(
        context: str | None = Provide[cloud_context],
    ) -> BentoCloudClient:
        from ..cloud import BentoCloudClient

        return BentoCloudClient.for_context(context)


BentoMLContainer = _BentoMLContainerClass()
load_config()
