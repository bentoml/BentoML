from __future__ import annotations

from typing import Annotated
from typing import Any
from typing import Dict
from typing import List
from typing import Sequence
from typing import Union

from annotated_types import Ge
from annotated_types import Gt
from annotated_types import Le
from pydantic import IPvAnyAddress
from pydantic import TypeAdapter
from typing_extensions import Literal
from typing_extensions import Required
from typing_extensions import TypedDict

Posint = Annotated[int, Gt(0)]


class EnvSchema(TypedDict, total=False):
    name: Required[str]
    required: bool
    default: Any


class TrafficSchema(TypedDict, total=False):
    timeout: Posint
    max_concurrency: Posint
    external_queue: bool  # bentocloud only


class ResourceSchema(TypedDict, total=False):
    num_cpus: Posint
    memory: str
    num_gpus: Posint
    gpu_type: str


class _IndividualWorkerSchema(TypedDict):
    gpus: Posint | list[Posint]


WorkerSchema = Union[Posint, List[_IndividualWorkerSchema]]


class MetricDuration(TypedDict, total=False):
    buckets: list[float]
    min: Annotated[float, Gt(0)]
    max: Annotated[float, Gt(0)]
    factor: Annotated[float, Gt(1.0)]


class MetricSchema(TypedDict, total=False):
    enabled: bool
    namespace: str
    duration: MetricDuration


class AccessLoggingSchema(TypedDict, total=False):
    enabled: bool
    request_content_length: bool
    request_content_type: bool
    response_content_length: bool
    response_content_type: bool
    format: TypedDict(
        "AccessLoggingFormat", {"trace_id": str, "span_id": str}, total=False
    )  # type: ignore


class SSLSchema(TypedDict, total=False):
    enabled: bool
    certfile: str
    keyfile: str
    keyfile_password: str
    ca_certs: str
    version: Posint
    cert_reqs: int
    ciphers: str


class HTTPCorsSchema(TypedDict, total=False):
    enabled: bool
    access_control_allow_origins: str | list[str]
    access_control_allow_credentials: bool
    access_control_allow_methods: str | list[str]
    access_control_allow_headers: str | list[str]
    access_control_allow_origin_regex: str
    access_control_max_age: int
    access_control_expose_headers: str | list[str]


class HTTPSchema(TypedDict, total=False):
    host: IPvAnyAddress
    port: int
    cors: HTTPCorsSchema
    response: TypedDict("HTTPResponseSchema", {"trace_id": bool}, total=False)  # type: ignore


class HostPortSchema(TypedDict):
    host: IPvAnyAddress
    port: Posint


class EnabledSchema(TypedDict):
    enabled: bool


class GRPCSchema(TypedDict, total=False):
    host: IPvAnyAddress
    port: Posint
    max_concurrent_streams: int
    maximum_concurrent_rpcs: int
    metrics: HostPortSchema
    reflection: EnabledSchema
    channelz: EnabledSchema
    max_message_length: int


class RunnerProbeSchema(TypedDict, total=False):
    enabled: bool
    timeout: Posint
    period: Posint


class MonitoringSchema(TypedDict, total=False):
    enabled: bool
    type: str
    options: dict[str, Any]


class ZipkinSchema(TypedDict, total=False):
    endpoint: str
    local_node_ipv4: IPvAnyAddress | int
    local_node_ipv6: IPvAnyAddress | int
    local_node_port: Posint


class JaegerSchema(TypedDict, total=False):
    protocol: Literal["grpc", "thrift"]
    collector_endpoint: str
    grpc: TypedDict("JaegerGRPCSchema", {"insecure": bool}, total=False)  # type: ignore
    thrift: TypedDict(
        "JaegerThriftSchema",
        {
            "agent_host_name": str,
            "agent_port": Posint,
            "udp_split_oversized_batches": bool,
        },
        total=False,
    )  # type: ignore


class OTLPSchema(TypedDict, total=False):
    protocol: Literal["grpc", "http"]
    endpoint: str
    compression: Literal["gzip", "none", "deflate"]
    http: TypedDict(
        "OTLPHTTPSchema",
        {"headers": Dict[str, str], "certificate_file": str},
        total=False,
    )  # type: ignore
    grpc: TypedDict(
        "OTLPGRPCSchema",
        {"insecure": bool, "headers": Sequence[Sequence[str]]},
        total=False,
    )  # type: ignore


class TracingSchema(TypedDict, total=False):
    exporter_type: Literal["zipkin", "jaeger", "otlp", "in_memory"]
    sample_rate: Annotated[float, Ge(0.0), Le(1.0)]
    timeout: Posint
    max_tag_value_length: Posint
    excluded_urls: list[str] | str
    zipkin: ZipkinSchema
    jaeger: JaegerSchema
    otlp: OTLPSchema


class LoggingSchema(TypedDict, total=False):
    access: AccessLoggingSchema


class Schema(TypedDict, total=False):
    envs: list[EnvSchema]
    traffic: TrafficSchema
    backlog: Annotated[int, Ge(64)]
    max_runner_connections: Posint
    resources: ResourceSchema
    workers: WorkerSchema
    metrics: MetricSchema
    logging: LoggingSchema
    ssl: SSLSchema
    http: HTTPSchema
    grpc: GRPCSchema
    runner_probe: RunnerProbeSchema
    tracing: TracingSchema
    monitoring: MonitoringSchema


schema_type = TypeAdapter(Schema)


def validate(data: Schema) -> Schema:
    return schema_type.validate_python(data)
