from __future__ import annotations

import re
import typing as t

import schema as s

from ..helpers import depth
from ..helpers import ensure_range
from ..helpers import rename_fields
from ..helpers import ensure_larger_than
from ..helpers import is_valid_ip_address
from ..helpers import ensure_iterable_type
from ..helpers import validate_tracing_type
from ..helpers import validate_otlp_protocol
from ..helpers import ensure_larger_than_zero
from ...utils.metrics import DEFAULT_BUCKET
from ...utils.unflatten import unflatten

__all__ = ["SCHEMA", "migration"]

TRACING_CFG = {
    "exporter_type": s.Or(s.And(str, s.Use(str.lower), validate_tracing_type), None),
    "sample_rate": s.Or(s.And(float, ensure_range(0, 1)), None),
    "timeout": s.Or(s.And(int, ensure_larger_than_zero), None),
    "max_tag_value_length": s.Or(int, None),
    "excluded_urls": s.Or([str], str, None),
    "zipkin": {
        "endpoint": s.Or(str, None),
        "local_node_ipv4": s.Or(s.Or(s.And(str, is_valid_ip_address), int), None),
        "local_node_ipv6": s.Or(s.Or(s.And(str, is_valid_ip_address), int), None),
        "local_node_port": s.Or(s.And(int, ensure_larger_than_zero), None),
    },
    "jaeger": {
        "protocol": s.Or(
            s.And(str, s.Use(str.lower), lambda d: d in ["thrift", "grpc"]),
            None,
        ),
        "collector_endpoint": s.Or(str, None),
        "thrift": {
            "agent_host_name": s.Or(str, None),
            "agent_port": s.Or(int, None),
            "udp_split_oversized_batches": s.Or(bool, None),
        },
        "grpc": {
            "insecure": s.Or(bool, None),
        },
    },
    "otlp": {
        "protocol": s.Or(s.And(str, s.Use(str.lower), validate_otlp_protocol), None),
        "endpoint": s.Or(str, None),
        "compression": s.Or(
            s.And(str, lambda d: d in {"gzip", "none", "deflate"}), None
        ),
        "http": {
            "certificate_file": s.Or(str, None),
            "headers": s.Or(dict, None),
        },
        "grpc": {
            "insecure": s.Or(bool, None),
            "headers": s.Or(lambda d: isinstance(d, t.Sequence), None),
        },
    },
}
_API_SERVER_CONFIG = {
    "workers": s.Or(s.And(int, ensure_larger_than_zero), None),
    "timeout": s.And(int, ensure_larger_than_zero),
    "backlog": s.And(int, ensure_larger_than(64)),
    "max_runner_connections": s.And(int, ensure_larger_than_zero),
    "metrics": {
        "enabled": bool,
        "namespace": str,
        s.Optional("duration"): {
            s.Optional("buckets", default=DEFAULT_BUCKET): s.Or(
                s.And(list, ensure_iterable_type(float)), None
            ),
            s.Optional("min"): s.Or(s.And(float, ensure_larger_than_zero), None),
            s.Optional("max"): s.Or(s.And(float, ensure_larger_than_zero), None),
            s.Optional("factor"): s.Or(s.And(float, ensure_larger_than(1.0)), None),
        },
    },
    "logging": {
        "access": {
            "enabled": bool,
            "request_content_length": s.Or(bool, None),
            "request_content_type": s.Or(bool, None),
            "response_content_length": s.Or(bool, None),
            "response_content_type": s.Or(bool, None),
            "format": {
                "trace_id": str,
                "span_id": str,
            },
        },
    },
    "http": {
        "host": s.And(str, is_valid_ip_address),
        "port": s.And(int, ensure_larger_than_zero),
        "cors": {
            "enabled": bool,
            "access_control_allow_origins": s.Or([str], str, None),
            "access_control_allow_origin_regex": s.Or(
                s.And(str, s.Use(re.compile)), None
            ),
            "access_control_allow_credentials": s.Or(bool, None),
            "access_control_allow_headers": s.Or([str], str, None),
            "access_control_allow_methods": s.Or([str], str, None),
            "access_control_max_age": s.Or(int, None),
            "access_control_expose_headers": s.Or([str], str, None),
        },
    },
    "grpc": {
        "host": s.And(str, is_valid_ip_address),
        "port": s.And(int, ensure_larger_than_zero),
        "metrics": {
            "port": s.And(int, ensure_larger_than_zero),
            "host": s.And(str, is_valid_ip_address),
        },
        "reflection": {"enabled": bool},
        "channelz": {"enabled": bool},
        "max_concurrent_streams": s.Or(int, None),
        "max_message_length": s.Or(int, None),
        "maximum_concurrent_rpcs": s.Or(int, None),
    },
    s.Optional("ssl"): {
        "enabled": bool,
        s.Optional("certfile"): s.Or(str, None),
        s.Optional("keyfile"): s.Or(str, None),
        s.Optional("keyfile_password"): s.Or(str, None),
        s.Optional("version"): s.Or(s.And(int, ensure_larger_than_zero), None),
        s.Optional("cert_reqs"): s.Or(int, None),
        s.Optional("ca_certs"): s.Or(str, None),
        s.Optional("ciphers"): s.Or(str, None),
    },
    "runner_probe": {
        "enabled": bool,
        "timeout": int,
        "period": int,
    },
}
_RUNNER_CONFIG = {
    s.Optional("batching"): {
        s.Optional("enabled"): bool,
        s.Optional("max_batch_size"): s.And(int, ensure_larger_than_zero),
        s.Optional("max_latency_ms"): s.And(int, ensure_larger_than_zero),
    },
    # NOTE: there is a distinction between being unset and None here; if set to 'None'
    # in configuration for a specific runner, it will override the global configuration.
    s.Optional("resources"): s.Or({s.Optional(str): object}, lambda s: s == "system", None),  # type: ignore (incomplete schema typing)
    s.Optional("logging"): {
        s.Optional("access"): {
            s.Optional("enabled"): bool,
            s.Optional("request_content_length"): s.Or(bool, None),
            s.Optional("request_content_type"): s.Or(bool, None),
            s.Optional("response_content_length"): s.Or(bool, None),
            s.Optional("response_content_type"): s.Or(bool, None),
        },
    },
    s.Optional("metrics"): {
        "enabled": bool,
        "namespace": str,
    },
    s.Optional("timeout"): s.And(int, ensure_larger_than_zero),
}
SCHEMA = s.Schema(
    {
        s.Optional("version", default=1): s.And(int, lambda v: v == 1),
        "api_server": _API_SERVER_CONFIG,
        "runners": {
            **_RUNNER_CONFIG,
            s.Optional(str): _RUNNER_CONFIG,
        },
        "tracing": TRACING_CFG,
        s.Optional("monitoring"): {
            "enabled": bool,
            s.Optional("type"): s.Or(str, None),
            s.Optional("options"): s.Or(dict, None),
        },
    }
)


def migration(*, override_config: dict[str, t.Any]):
    # We will use a flattened config to make it easier to migrate,
    # Then we will convert it back to a nested config.
    if depth(override_config) > 1:
        raise ValueError("'override_config' must be a flattened dictionary.") from None

    if "version" not in override_config:
        override_config["version"] = 1

    # First we migrate api_server field
    # 1. remove api_server.max_request_size (deprecated)
    rename_fields(
        override_config, current="api_server.max_request_size", remove_only=True
    )

    # 2. migrate api_server.[host|port] -> api_server.http.[host|port]
    for f in ["host", "port"]:
        rename_fields(
            override_config,
            current=f"api_server.{f}",
            replace_with=f"api_server.http.{f}",
        )

    # 3. migrate api_server.cors.[access_control_*] -> api_server.http.cors.[*]
    rename_fields(
        override_config,
        current="api_server.cors.enabled",
        replace_with="api_server.http.cors.enabled",
    )
    for f in [
        "allow_origin",
        "allow_credentials",
        "allow_headers",
        "allow_methods",
        "max_age",
        "expose_headers",
    ]:
        rename_fields(
            override_config,
            current=f"api_server.cors.access_control_{f}",
            replace_with=f"api_server.http.cors.access_control_{f}",
        )

    rename_fields(
        override_config,
        current="api_server.http.cors.access_control_allow_origin",
        replace_with="api_server.http.cors.access_control_allow_origins",
    )

    # 4. if ssl is present, in version 2 we introduce a api_server.ssl.enabled field to determine
    # whether user want to enable SSL.
    if len([f for f in override_config if f.startswith("api_server.ssl")]) != 0:
        override_config["api_server.ssl.enabled"] = True

    # 5. migrate all tracing fields to api_server.tracing
    # 5.1. migrate tracing.type -> api_server.tracing.expoerter_type
    rename_fields(
        override_config,
        current="tracing.type",
        replace_with="tracing.exporter_type",
    )
    # 5.2. for Zipkin and OTLP, migrate tracing.[exporter].url -> api_server.tracing.[exporter].endpoint
    for exporter in ["zipkin", "otlp"]:
        rename_fields(
            override_config,
            current=f"tracing.{exporter}.url",
            replace_with=f"tracing.{exporter}.endpoint",
        )
    # 5.3. For Jaeger, migrate tracing.jaeger.[address|port] -> api_server.tracing.jaeger.thrift.[agent_host_name|agent_port]
    rename_fields(
        override_config,
        current="tracing.jaeger.address",
        replace_with="tracing.jaeger.thrift.agent_host_name",
    )
    rename_fields(
        override_config,
        current="tracing.jaeger.port",
        replace_with="tracing.jaeger.thrift.agent_port",
    )
    # we also need to choose which protocol to use for jaeger.
    if (
        len(
            [
                f
                for f in override_config
                if f.startswith("api_server.tracing.jaeger.thrift")
            ]
        )
        != 0
    ):
        override_config["tracing.jaeger.protocol"] = "thrift"
    # 6. Last but not least, moving logging.formatting.* -> api_server.logging.access.format.*
    for f in ["trace_id", "span_id"]:
        rename_fields(
            override_config,
            current=f"logging.formatting.{f}_format",
            replace_with=f"api_server.logging.access.format.{f}",
        )
    return unflatten(override_config)
